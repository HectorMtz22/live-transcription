"""Engine shutdown must tell the Qwen translator it is stopping BEFORE the
thread pools drain with wait=True.

Ctrl+C delivers SIGINT to the whole foreground process group, so the Qwen
child dies. If the pools then drain an in-flight translate/retranslate, the
watchdog would see a dead child and spawn a fresh one (reloading the 8B model)
mid-shutdown. engine.stop() must call translator.begin_shutdown() first so the
watchdog stays quiet.
"""


def test_engine_stop_begins_shutdown_before_pool_drain(
    patched_engine,
    fake_qwen_translator,
    fake_whisper_result,
):
    # Shared timeline recorded by the translator spy AND the pool-drain spies,
    # so we can assert their relative order.
    order = []
    base_cls = type(fake_qwen_translator())

    class SpyQwen(base_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._stopping = False

        def begin_shutdown(self):
            order.append("begin_shutdown")
            super().begin_shutdown()

        def stop(self):
            order.append("stop")
            return super().stop()

    translator = SpyQwen(target_lang="es", default="hola")

    engine, _listener = patched_engine(
        translator=translator,
        translate_langs=["en"],
        target_lang="es",
        whisper_result=fake_whisper_result("hi", lang="en"),
    )
    engine.start()

    # Wrap each pool's shutdown so the drain records its own position in the
    # timeline. This pins the load-bearing invariant (begin_shutdown BEFORE the
    # drain) directly, rather than inferring it from source order — a refactor
    # moving the drains ahead of begin_shutdown would then fail this test.
    def _spy_pool_shutdown(pool, label):
        original = pool.shutdown

        def wrapper(*args, **kwargs):
            order.append(label)
            return original(*args, **kwargs)

        pool.shutdown = wrapper

    _spy_pool_shutdown(engine._transcription_pool, "drain")
    _spy_pool_shutdown(engine._translation_pool, "drain")

    engine.stop()

    assert translator._stopping is True
    assert "begin_shutdown" in order

    first_drain = next(i for i, ev in enumerate(order) if ev == "drain")
    # begin_shutdown() must precede the FIRST pool drain...
    assert order.index("begin_shutdown") < first_drain
    # ...and translator.stop() must come only after the drains have run.
    assert order.index("stop") > first_drain
