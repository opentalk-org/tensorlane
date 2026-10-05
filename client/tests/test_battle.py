from concurrent.futures import ThreadPoolExecutor
import time
from unittest.mock import patch

from fixture_transforms import identify_worker
from pipeline_fixture import PipelineCase


class BattleTests(PipelineCase):
    def consume(self, rank, stream, delay=0):
        with self.attach(rank) as lane:
            with lane.batches(stream, timeout=20) as reader:
                result = []
                for batch in reader:
                    time.sleep(delay)
                    result.append(
                        (
                            batch.batch_id,
                            batch.query_batch_idx,
                            [sample["value"][0].item() for sample in batch],
                        )
                    )
                return result

    def check_distributed(self, ranks, workers, factor, delay=0):
        self.start(
            ranks=ranks,
            workers=workers,
            factor=factor,
            transform_fn=identify_worker,
            performance_metrics=False,
        )
        with ThreadPoolExecutor(max_workers=ranks * len(self.service.streams)) as pool:
            futures = {
                (rank, stream): pool.submit(
                    self.consume, rank, stream, delay if rank == 0 else 0
                )
                for stream in self.service.streams
                for rank in range(ranks)
            }
            for (rank, stream), future in futures.items():
                result = future.result(timeout=45)
                offset = 1000 * list(self.service.streams).index(stream)
                self.assertEqual(
                    result,
                    [
                        (index, index * 4 + 7, [index + offset] * (index % 3 + 1))
                        for index in range(rank, self.service.streams[stream], ranks)
                    ],
                )

    def test_empty_streams_end_on_every_rank(self):
        self.service.streams = dict.fromkeys(self.service.streams, 0)
        self.check_distributed(ranks=4, workers=3, factor=1)

    def test_more_ranks_than_batches(self):
        self.service.streams.update(training=2, validation=1, evaluation=0)
        self.check_distributed(ranks=4, workers=3, factor=1)

    def test_slow_rank_and_out_of_order_workers_preserve_all_samples(self):
        self.service.streams.update(training=97, validation=35, evaluation=17)
        self.check_distributed(ranks=4, workers=4, factor=3, delay=0.01)

    def test_single_worker_with_many_credits_preserves_tensor_values(self):
        self.service.streams.update(training=129, validation=7, evaluation=1)
        self.check_distributed(ranks=2, workers=1, factor=32)

    def test_dot_stream_names_are_rejected_before_batch_requests(self):
        for name in (".", ".."):
            with self.subTest(name=name):
                self.service.streams = {name: 1}
                self.service.requests = {name: []}
                self.service.config["queries"] = {name: {"sql": "SELECT ..."}}
                with self.assertRaisesRegex(RuntimeError, "stream name must not be"):
                    self.start(workers=1, performance_metrics=False)
                self.assertEqual(self.service.requests[name], [])

    def test_dot_asset_names_are_rejected_before_download(self):
        for name in (".", ".."):
            with self.subTest(name=name):
                self.service.assets = {name: (None, b"weights")}
                with self.assertRaisesRegex(
                    RuntimeError, "HTTP path segment must not be"
                ):
                    self.start(workers=1, performance_metrics=False)
                self.assertEqual(self.service.asset_requests, [])

    def test_busy_close_reports_success_despite_worker_disconnects(self):
        self.service.streams.update(training=129)
        self.start(
            workers=1,
            factor=32,
            transform_fn=identify_worker,
            performance_metrics=False,
        )
        stop_python = self.daemon._stopped.set

        def delayed_stop():
            stop_python()
            # Allow workers to disconnect before close continues.
            time.sleep(0.5)

        with patch.object(self.daemon._stopped, "set", side_effect=delayed_stop):
            self.daemon.close()
        self.assertFalse(self.service.end_requests[-1].failed)
