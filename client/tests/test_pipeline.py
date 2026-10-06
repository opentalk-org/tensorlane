from __future__ import annotations
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from collections import UserDict
from unittest.mock import patch
import time
import uuid
import tensorlane
from fixture_transforms import transform, collate, identify_worker
from pipeline_fixture import PipelineCase, Fixture, Authorization, wait_for


def read_rank(run_id, rank, root, output, stream="training"):
    try:
        with tensorlane.init(
            run_id, ranks=2, rank=rank, start_daemon=False, ipc_dir=root, timeout=20
        ) as lane:
            with lane.batches(stream, timeout=20) as batches:
                result = [
                    (batch.batch_id, batch.samples[0]["value"].tolist())
                    for batch in batches
                ]
            output.put((rank, result, None))
    except Exception as error:
        output.put((rank, None, str(error)))


class PipelineTests(PipelineCase):
    def test_two_runs_share_batches_without_changing_process_authentication(self):
        key = multiprocessing.current_process().authkey
        self.service.blob_size = 1024 * 1024
        first = self.start(workers=1, performance_metrics=False)
        with tensorlane.init(
            str(uuid.uuid4()),
            transform,
            num_workers=1,
            addr=f"localhost:{self.service.port}",
            ipc_dir=self.temp.name,
            performance_metrics=False,
            timeout=20,
        ) as second:
            self.assertEqual(multiprocessing.current_process().authkey, key)

            def consume(lane):
                with lane.batches("training") as reader:
                    for index, batch in enumerate(reader):
                        self.assertEqual(batch.batch_id, index)
                        for sample in batch:
                            self.assertEqual(sample["value"][0].item(), index)
                            self.assertTrue(sample["value"].is_shared())
                            self.assertEqual(sample["nested"][1][0], b"x" * 1024 * 1024)
                    return index + 1

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(consume, lane) for lane in (first, second)]
                self.assertEqual(
                    [future.result(timeout=30) for future in futures], [5, 5]
                )

    def test_transformed_and_collated_tensors_use_shared_memory(self):
        self.start(collate_fn=collate)
        with self.daemon.batches("training") as reader:
            batch = next(reader)
            self.assertTrue(batch.samples[0]["value"].is_shared())
            self.assertTrue(batch.data["values"].is_shared())

    def test_callbacks_accept_non_dict_mappings(self):
        self.start(
            transform_fn=UserDict({"training": transform}),
            collate_fn=UserDict({"training": collate}),
        )
        with self.daemon.batches("training") as reader:
            batches = list(reader)
        self.assertEqual(len(batches), self.service.streams["training"])
        for index, batch in enumerate(batches):
            self.assertEqual(batch.data["values"][:, 0].tolist(), [index] * len(batch))
        with self.daemon.batches("evaluation") as reader:
            self.assertIsInstance(next(reader).samples[0], tensorlane.RawSample)

    def test_api_key_from_environment_authenticates_every_endpoint(self):
        key = "0123456789abcdef0123456789abcdef"
        auth = Authorization(key)
        self.service.close()
        self.service = Fixture(auth)
        self.service.assets = {"model": (str(uuid.uuid4()), b"input")}
        with patch.dict("os.environ", {"TENSORLANE_API_KEY": key}):
            self.start()
        self.daemon.metric(1, "loss", 0.5)
        self.daemon.save_asset("model", self.file())
        self.daemon.flush()
        with self.daemon.batches("training") as batches:
            list(batches)
        self.daemon.close()
        self.assertEqual(
            auth.methods,
            {
                "init",
                "batches",
                "inputs",
                "model",
                "uploads",
                "metrics",
                "end",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "Unauthorized"):
            self.start()

    def test_config_callbacks_and_three_streams(self):
        self.start(collate_fn={"training": collate})
        self.assertEqual(self.daemon.config, self.service.config)
        self.assertEqual(self.daemon.streams, tuple(self.service.streams))
        with self.assertRaises(TypeError):
            self.daemon.batches()
        for stream, count in self.service.streams.items():
            with self.daemon.batches(stream) as reader:
                batches = list(reader)
            self.assertEqual(len(batches), count)
            for index, batch in enumerate(batches):
                self.assertEqual(
                    (batch.stream, batch.batch_id, batch.query_batch_idx),
                    (stream, index, index * 4 + 7),
                )
                self.assertEqual(len(batch), index % 3 + 1)
                if stream == "training":
                    self.assertEqual(batch.data["values"].shape[0], len(batch))
                else:
                    self.assertEqual(batch.data, batch.samples)

    def test_runtime_parameters_from_json_are_shared_with_followers(self):
        self.service.config["tensorlane"].update(
            ranks=2, num_workers=2, prefetch_factor=1
        )
        self.service.config["app"].update(
            ranks=99, num_workers="custom", prefetch_factor=0
        )
        self.daemon = tensorlane.init(
            self.run_id, addr=f"localhost:{self.service.port}", ipc_dir=self.temp.name
        )
        self.assertEqual(
            (
                self.daemon.rank,
                self.daemon.ranks,
                self.daemon.num_workers,
                self.daemon.prefetch_factor,
            ),
            (0, 2, 2, 1),
        )
        self.assertEqual(len(self.daemon._processes), 3)
        with tensorlane.init(self.run_id, rank=1, ipc_dir=self.temp.name) as follower:
            self.assertEqual(
                (
                    follower.rank,
                    follower.ranks,
                    follower.num_workers,
                    follower.prefetch_factor,
                ),
                (1, 2, 2, 1),
            )

    def test_explicit_runtime_parameters_override_json(self):
        self.service.config["tensorlane"].update(
            ranks=2, num_workers=2, prefetch_factor=1
        )
        self.start(workers=1)
        self.assertEqual(
            (self.daemon.ranks, self.daemon.num_workers, self.daemon.prefetch_factor),
            (1, 1, 2),
        )
        self.assertEqual(self.daemon.config, self.service.config)

    def test_existing_run_id_from_environment_needs_no_argument(self):
        self.service.config["tensorlane"].update(num_workers=1)
        with patch.dict("os.environ", {"TENSORLANE_RUN_ID": self.run_id}):
            self.daemon = tensorlane.init(
                addr=f"localhost:{self.service.port}", ipc_dir=self.temp.name
            )
        self.assertEqual(self.daemon.run_id, self.run_id)

    def test_raw_samples_and_mapping_defaults(self):
        self.start(transform_fn={"training": transform})
        with self.daemon.batches("evaluation") as reader:
            sample = next(reader).samples[0]
            self.assertIsInstance(sample, tensorlane.RawSample)
            self.assertEqual(sample.blobs["context"], b"arbitrary bytes")
            self.assertEqual(sample.metadata["position"], 2000)

    def test_init_and_asset_loading_do_not_fetch_streams(self):
        self.service.assets = {"model": (None, b"weights")}
        self.start(factor=1, workers=1, performance_metrics=False)
        self.assertEqual(self.daemon.asset("model").read_bytes(), b"weights")
        time.sleep(0.1)
        self.assertTrue(
            all(not requests for requests in self.service.requests.values())
        )
        with self.daemon.batches("evaluation") as reader:
            self.assertEqual(len(list(reader)), self.service.streams["evaluation"])
        self.assertEqual(self.service.requests["training"], [])
        self.assertEqual(self.service.requests["validation"], [])
        with self.daemon.batches("validation") as reader:
            self.assertEqual(len(list(reader)), self.service.streams["validation"])
        self.assertEqual(self.service.requests["training"], [])

    def test_other_streams_drain_while_training_is_full(self):
        self.start(factor=1, workers=3)
        with self.daemon.batches("training") as training:
            wait_for(lambda: len(self.service.requests["training"]) == 1)
            for name in ("validation", "evaluation"):
                with self.daemon.batches(name) as reader:
                    self.assertEqual(len(list(reader)), self.service.streams[name])
            self.assertEqual(len(self.service.requests["training"]), 1)
            self.assertEqual(len(list(training)), 5)

    def test_credit_budget_covers_unconsumed_batches(self):
        self.start(factor=1)
        with self.daemon.batches("training") as reader:
            wait_for(lambda: len(self.service.requests["training"]) == 1)
            time.sleep(0.1)
            self.assertEqual(len(self.service.requests["training"]), 1)
            next(reader)
            wait_for(lambda: len(self.service.requests["training"]) == 2)
            time.sleep(0.1)
            self.assertEqual(len(self.service.requests["training"]), 2)

    def test_oversized_prefetch_batches_drain_independent_streams(self):
        self.service.config["tensorlane"]["max_prefetch_memory_bytes"] = 3 * 1024 * 1024
        self.service.blob_size = 2 * 1024 * 1024
        self.start(factor=4, workers=3)
        received = {}
        with self.daemon.batches("training") as training:
            wait_for(lambda: bool(self.service.requests["training"]))
            for name in ("validation", "evaluation"):
                with self.daemon.batches(name) as reader:
                    received[name] = list(reader)
            received["training"] = list(training)
        for name, batches in received.items():
            self.assertEqual(
                [batch.batch_id for batch in batches],
                list(range(self.service.streams[name])),
            )
            for batch in batches:
                self.assertEqual(len(batch), batch.batch_id % 3 + 1)
                self.assertTrue(all(sample["value"].is_shared() for sample in batch))

    def test_multiple_workers_preserve_order(self):
        self.service.streams["training"] = 10
        self.start(factor=4, workers=3, transform_fn=identify_worker)
        with self.daemon.batches("training") as reader:
            batches = list(reader)
        self.assertEqual(
            [batch.samples[0]["value"][0].item() for batch in batches], list(range(10))
        )
        self.assertEqual(
            len({sample["value"][1].item() for batch in batches for sample in batch}), 3
        )

    def test_independent_ranks_attach_before_owner(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        processes = [
            context.Process(
                target=read_rank, args=(self.run_id, rank, self.temp.name, output)
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            self.start(ranks=2, workers=3)
            for _ in processes:
                rank, batches, error = output.get(timeout=30)
                self.assertIsNone(error)
                self.assertEqual(
                    [batch[0] for batch in batches], list(range(rank, 5, 2))
                )
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join()
            output.close()
