import json
import torch
from pipeline_fixture import PipelineCase, wait_for


class CheckpointResumeTests(PipelineCase):
    def test_same_run_resumes_saved_model_and_cursor_without_fetching_old_batches(self):
        self.start(workers=1, factor=4, performance_metrics=False)
        with (
            self.daemon.batches("training") as training,
            self.daemon.batches("validation") as validation,
        ):
            self.assertEqual(next(training).batch_id, 0)
            self.assertEqual(next(training).batch_id, 1)
            self.assertEqual(next(validation).batch_id, 0)
            wait_for(
                lambda: any(row.index >= 3 for row in self.service.requests["training"])
            )
            self.daemon.save_asset(
                "model", {"weight": torch.tensor([7.0])}, kind="checkpoint"
            )
            saved = self.service.saved[-1][0]
            self.assertEqual(saved.step, 1)
            self.assertEqual(
                json.loads(saved.metadata_json)["_tensorlane"]["next_batches"],
                {"training": 2, "validation": 1, "evaluation": 0},
            )
            self.assertEqual(next(training).batch_id, 2)
        self.daemon.close()
        self.service.requests = {name: [] for name in self.service.streams}
        self.start(workers=1, performance_metrics=False)
        self.assertEqual(self.daemon.run_id, self.run_id)
        self.assertEqual(
            torch.load(self.daemon.asset("model"), weights_only=True)["weight"].item(),
            7,
        )
        with self.daemon.batches("training") as training:
            self.assertEqual([batch.batch_id for batch in training], [2, 3, 4])
        with self.daemon.batches("validation") as validation:
            self.assertEqual([batch.batch_id for batch in validation], [1, 2])
        self.assertTrue(
            all(row.index >= 2 for row in self.service.requests["training"])
        )
        self.daemon.save_asset(
            "model", {"weight": torch.tensor([9.0])}, kind="checkpoint"
        )
        self.daemon.close()
        self.start(workers=1, performance_metrics=False)
        with self.daemon.batches("training") as training:
            self.assertEqual(list(training), [])
        self.assertEqual(
            torch.load(self.daemon.asset("model"), weights_only=True)["weight"].item(),
            9,
        )

    def test_checkpoint_uses_rank_progress_and_preserves_explicit_step(self):
        self.start(ranks=2, workers=1, performance_metrics=False)
        with self.attach(rank=1) as follower:
            with (
                self.daemon.batches("training") as first,
                follower.batches("training") as second,
            ):
                self.assertEqual(next(first).batch_id, 0)
                self.daemon.save_asset("model", {}, kind="checkpoint", step=99)
                saved = self.service.saved[-1][0]
                self.assertEqual(saved.step, 99)
                self.assertEqual(
                    json.loads(saved.metadata_json)["_tensorlane"]["next_batches"][
                        "training"
                    ],
                    1,
                )
                self.assertEqual(next(second).batch_id, 1)
                self.daemon.save_asset("model", {}, kind="checkpoint")
                self.assertEqual(
                    json.loads(self.service.saved[-1][0].metadata_json)["_tensorlane"][
                        "next_batches"
                    ]["training"],
                    2,
                )
        self.daemon.close()
        self.start(ranks=2, workers=1, performance_metrics=False)
        with self.attach(rank=1) as follower:
            with (
                self.daemon.batches("training") as first,
                follower.batches("training") as second,
            ):
                self.assertEqual(next(first).batch_id, 2)
                self.assertEqual(next(second).batch_id, 3)
