"""Custom hooks for tf.Estimator/MonitoredSession."""
import tensorflow as tf


class SummarySaverHookWithProfile(tf.train.SummarySaverHook):
    """Modified version of the standard SummarySaverHook including profiling.

    Use this instead of ProfilerHook, which just writes JSON files to disk you
    would have to look at some other way. This hook uses the Tensorboard
    profiling functionality instead.
    """

    def __init__(self, save_steps=None, save_secs=None, profile_steps=None,
                 profile_secs=None, output_dir=None, summary_writer=None,
                 scaffold=None, summary_op=None):
        """Initialize object. See SummarySaverHook.

        We just set up a second timer for profiling.
        """
        super().__init__(save_steps, save_secs, output_dir, summary_writer,
                         scaffold, summary_op)
        self._profile_timer = tf.train.SecondOrStepTimer(
            every_secs=profile_secs, every_steps=profile_steps)

    def before_run(self, run_context):
        """Check if summary and/or profiling is requested this run."""
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self._global_step_tensor}
        if self._request_summary:
            if self._get_summary_op() is not None:
                requests["summary"] = self._get_summary_op()

        self._request_profile = (
            self._next_step is None or
            self._profile_timer.should_trigger_for_step(self._next_step))
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                if self._request_summary else None)

        return tf.train.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        """Add summaries/profiling if requested."""
        _ = run_context
        if not self._summary_writer:
            return

        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._next_step is None or self._request_summary or \
                self._request_profile:
            global_step = run_context.session.run(self._global_step_tensor)

        if self._next_step is None:
            self._summary_writer.add_session_log(
                tf.SessionLog(status=tf.SessionLog.START), global_step)
        if self._request_summary:
            self._timer.update_last_triggered_step(global_step)
            if "summary" in run_values.results:
                for summary in run_values.results["summary"]:
                    self._summary_writer.add_summary(summary, global_step)

        if self._request_profile:
            self._profile_timer.update_last_triggered_step(global_step)
            self._summary_writer.add_run_metadata(run_values.run_metadata,
                                                  "step{}".format(global_step),
                                                  global_step=global_step)
            print("Added profiling for step {}.".format(global_step))
        self._next_step = global_step + 1
