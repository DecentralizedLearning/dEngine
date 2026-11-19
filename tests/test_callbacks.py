import unittest
from unittest.mock import MagicMock, patch

from dengine.callbacks import CallbackList, DummyCallback, PeriodicCallback, callback_factory
from dengine.callbacks.decorators import BUILTIN_CALLBACKS, register_callback
from dengine.config import DynamicModuleConfigBase
from dengine.interfaces import ClientCallbackInterface


# (hook name, positional args, keyword args) for every public ``on_*`` hook.
HOOK_CALLS = [
    ('on_synchronization_start', (1.0,), {}),
    ('on_synchronization_end', (1.0, ['msg']), {}),
    ('on_aggregation_start', (1.0,), {}),
    ('on_aggregation_end', (1.0,), {}),
    ('on_local_training_start', (1.0,), {}),
    ('on_local_training_end', (1.0,), {'loss': 0.5}),
    ('on_training_batch_start', (3,), {}),
    ('on_training_batch_end', (3,), {'loss': 0.5}),
    ('on_training_epoch_start', (2,), {}),
    ('on_training_epoch_end', (2,), {'loss': 0.5}),
    ('on_test_inference_start', (1.0,), {}),
    ('on_test_inference_end', (1.0,), {'accuracy': 0.9}),
]

# on_* hook -> redefinable hook that PeriodicCallback forwards to.
REDEFINITIONS = {hook: hook.removeprefix('on_') for hook, _, _ in HOOK_CALLS}

# A target that is neither registered nor importable is reported through the generic
# module loader, which raises one of these depending on the shape of the name.
UNRESOLVED_TARGET_ERRORS = (ModuleNotFoundError, AttributeError)

# Shared trace written by the doubles below. Tests clear it in setUp.
TRACE = []


class ListenerDouble:
    """Records the public ``on_*`` calls it receives as (label, hook, args, kwargs)."""

    def __init__(self, label):
        self.label = label


def _make_listener_hook(hook):
    def method(self, *args, **kwargs):
        TRACE.append((self.label, hook, args, kwargs))
    return method


for _hook, _, _ in HOOK_CALLS:
    setattr(ListenerDouble, _hook, _make_listener_hook(_hook))


class RaisingListener(ListenerDouble):
    def on_local_training_start(self, *args, **kwargs):
        raise RuntimeError('boom')


class RecordingCallback(PeriodicCallback):
    """Records every redefinable hook it receives as (hook name, args, kwargs)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []


def _make_recording_hook(name):
    def hook(self, *args, **kwargs):
        self.calls.append((name, args, kwargs))
    return hook


for _name in REDEFINITIONS.values():
    setattr(RecordingCallback, _name, _make_recording_hook(_name))


class TraceCallback(PeriodicCallback):
    """Importable callback that traces its constructor inputs and the hooks it receives."""

    KIND = 'trace'

    def __init__(self, client, scenario, test_data, experiment_configuration, label='cb', **periods):
        super().__init__(client, scenario, test_data, experiment_configuration, **periods)
        self.label = label
        TRACE.append(('init', self.KIND, label, client, scenario, test_data, experiment_configuration))


def _make_trace_hook(name):
    def hook(self, *args, **kwargs):
        TRACE.append((self.label, name, args, kwargs))
    return hook


for _name in REDEFINITIONS.values():
    setattr(TraceCallback, _name, _make_trace_hook(_name))


class DefaultNamedTrace(TraceCallback):
    KIND = 'default'


class ExplicitNamedTrace(TraceCallback):
    KIND = 'explicit'


def make_callback(cls=RecordingCallback, **periods):
    return cls(
        client=MagicMock(),
        scenario=MagicMock(),
        test_data=MagicMock(),
        experiment_configuration=MagicMock(),
        **periods
    )


class IsolatedRegistryTestCase(unittest.TestCase):
    """Restores BUILTIN_CALLBACKS after every test, whatever the test order."""

    def setUp(self):
        TRACE.clear()
        self.addCleanup(TRACE.clear)
        patcher = patch.dict(BUILTIN_CALLBACKS)
        patcher.start()
        self.addCleanup(patcher.stop)


class RegisterCallbackTest(IsolatedRegistryTestCase):
    def test_default_name_is_the_class_name(self):
        @register_callback()
        class DefaultNamed(RecordingCallback):
            pass

        self.assertIs(BUILTIN_CALLBACKS['DefaultNamed'], DefaultNamed)

    def test_explicit_name_replaces_the_class_name(self):
        @register_callback('explicit_name')
        class ExplicitlyNamed(RecordingCallback):
            pass

        self.assertIs(BUILTIN_CALLBACKS['explicit_name'], ExplicitlyNamed)
        self.assertNotIn('ExplicitlyNamed', BUILTIN_CALLBACKS)

    def test_decorator_returns_the_class_unchanged(self):
        class Plain(RecordingCallback):
            pass

        self.assertIs(register_callback()(Plain), Plain)

    def test_duplicate_default_name_is_rejected(self):
        @register_callback()
        class Duplicated(RecordingCallback):
            pass

        impostor = type('Duplicated', (RecordingCallback,), {})
        with self.assertRaisesRegex(ValueError, "'Duplicated'"):
            register_callback()(impostor)

        # The first registration is kept.
        self.assertIs(BUILTIN_CALLBACKS['Duplicated'], Duplicated)

    def test_duplicate_explicit_name_is_rejected(self):
        @register_callback('shared_name')
        class First(RecordingCallback):
            pass

        with self.assertRaisesRegex(ValueError, "'shared_name'"):
            @register_callback('shared_name')
            class Second(RecordingCallback):
                pass

        self.assertIs(BUILTIN_CALLBACKS['shared_name'], First)

    def test_explicit_name_cannot_shadow_a_default_name(self):
        @register_callback()
        class Taken(RecordingCallback):
            pass

        with self.assertRaises(ValueError):
            register_callback('Taken')(type('Other', (RecordingCallback,), {}))

        self.assertIs(BUILTIN_CALLBACKS['Taken'], Taken)

    def test_builtin_names_are_protected(self):
        builtin_name = next(iter(BUILTIN_CALLBACKS))
        before = BUILTIN_CALLBACKS[builtin_name]

        with self.assertRaises(ValueError):
            register_callback(builtin_name)(type('Imposter', (RecordingCallback,), {}))

        self.assertIs(BUILTIN_CALLBACKS[builtin_name], before)

    def test_builtin_registry_holds_callback_classes(self):
        self.assertTrue(BUILTIN_CALLBACKS)
        for name, cls in BUILTIN_CALLBACKS.items():
            with self.subTest(name=name):
                self.assertTrue(issubclass(cls, ClientCallbackInterface))


class CallbackFactoryTest(IsolatedRegistryTestCase):
    def setUp(self):
        super().setUp()
        register_callback()(DefaultNamedTrace)
        register_callback('factory_explicit')(ExplicitNamedTrace)
        self.client = MagicMock()
        self.scenario = MagicMock()
        self.test_data = MagicMock()
        self.experiment_configuration = MagicMock()

    def build(self, configuration):
        return callback_factory(
            self.client,
            self.scenario,
            configuration,
            test_data=self.test_data,
            experiment_configuration=self.experiment_configuration,
        )

    def config(self, target, **arguments):
        return DynamicModuleConfigBase(target=target, arguments=arguments or None)

    def inits(self):
        return [entry for entry in TRACE if entry[0] == 'init']

    def test_resolves_default_and_explicit_names_in_config_order(self):
        callbacks = self.build([
            self.config('factory_explicit', label='a'),
            self.config('DefaultNamedTrace', label='b'),
        ])

        self.assertIsInstance(callbacks, CallbackList)
        self.assertEqual([(kind, label) for _, kind, label, *_ in self.inits()],
                         [('explicit', 'a'), ('default', 'b')])

        TRACE.clear()
        callbacks.on_local_training_start(1.0)
        self.assertEqual(TRACE, [('a', 'local_training_start', (1.0,), {}),
                                 ('b', 'local_training_start', (1.0,), {})])

    def test_class_name_of_an_explicitly_named_callback_is_not_a_registry_key(self):
        with self.assertRaises(UNRESOLVED_TARGET_ERRORS):
            self.build([self.config('ExplicitNamedTrace')])

    def test_forwards_constructor_inputs(self):
        self.build([self.config('DefaultNamedTrace', label='x')])

        ((_, kind, label, client, scenario, test_data, experiment_configuration),) = self.inits()
        self.assertEqual((kind, label), ('default', 'x'))
        self.assertIs(client, self.client)
        self.assertIs(scenario, self.scenario)
        self.assertIs(test_data, self.test_data)
        self.assertIs(experiment_configuration, self.experiment_configuration)

    def test_configured_period_applies_to_dispatch(self):
        callbacks = self.build([self.config('DefaultNamedTrace', label='p', every_n_tests=2)])

        TRACE.clear()
        for time in (1.0, 2.0, 3.0, 4.0):
            callbacks.on_test_inference_end(time, accuracy=0.5)

        self.assertEqual(TRACE, [('p', 'test_inference_end', (2.0,), {'accuracy': 0.5}),
                                 ('p', 'test_inference_end', (4.0,), {'accuracy': 0.5})])

    def test_repeated_entries_are_dispatched_independently(self):
        config = self.config('DefaultNamedTrace', label='same')

        callbacks = self.build([config, config])

        self.assertEqual(len(self.inits()), 2)
        TRACE.clear()
        callbacks.on_aggregation_end(1.0)
        self.assertEqual(TRACE, [('same', 'aggregation_end', (1.0,), {})] * 2)

    def test_empty_configuration_yields_a_callback_list_that_ignores_every_hook(self):
        callbacks = self.build([])

        self.assertIsInstance(callbacks, CallbackList)
        for hook, args, kwargs in HOOK_CALLS:
            with self.subTest(hook=hook):
                getattr(callbacks, hook)(*args, **kwargs)
        self.assertEqual(TRACE, [])

    def test_unregistered_name_is_an_error(self):
        with self.assertRaises(UNRESOLVED_TARGET_ERRORS):
            self.build([self.config('NotRegisteredAnywhere')])

    def test_dotted_import_path_works_without_registration(self):
        self.assertNotIn('TraceCallback', BUILTIN_CALLBACKS)

        callbacks = self.build([self.config(f'{__name__}.TraceCallback', label='dotted')])

        TRACE.clear()
        callbacks.on_local_training_end(1.0, loss=0.1)
        self.assertEqual(TRACE, [('dotted', 'local_training_end', (1.0,), {'loss': 0.1})])

    def test_target_must_be_a_callback_class(self):
        with self.assertRaises(TypeError):
            self.build([self.config('collections.OrderedDict')])


class CallbackListDispatchTest(unittest.TestCase):
    def setUp(self):
        TRACE.clear()
        self.addCleanup(TRACE.clear)

    def test_every_hook_reaches_every_callback_in_order_with_unchanged_arguments(self):
        callbacks = CallbackList([ListenerDouble('first'), ListenerDouble('second')])

        for hook, args, kwargs in HOOK_CALLS:
            with self.subTest(hook=hook):
                TRACE.clear()
                getattr(callbacks, hook)(*args, **kwargs)
                self.assertEqual(TRACE, [('first', hook, args, kwargs), ('second', hook, args, kwargs)])

    def test_empty_list_ignores_every_hook(self):
        callbacks = CallbackList([])

        for hook, args, kwargs in HOOK_CALLS:
            with self.subTest(hook=hook):
                getattr(callbacks, hook)(*args, **kwargs)

        self.assertEqual(TRACE, [])

    def test_failing_callback_stops_dispatch_and_propagates(self):
        callbacks = CallbackList([ListenerDouble('before'), RaisingListener('failing'), ListenerDouble('after')])

        with self.assertRaisesRegex(RuntimeError, 'boom'):
            callbacks.on_local_training_start(1.0)

        self.assertEqual(TRACE, [('before', 'on_local_training_start', (1.0,), {})])


class PeriodicCallbackTest(unittest.TestCase):
    def fired(self, hook, period_name, period, values, extra_args=()):
        recorder = make_callback(**{period_name: period})
        for value in values:
            getattr(recorder, hook)(value, *extra_args)
        return [args[0] for _, args, _ in recorder.calls]

    def test_hooks_forward_to_their_redefinition_when_no_period_is_set(self):
        recorder = make_callback()

        for hook, args, kwargs in HOOK_CALLS:
            with self.subTest(hook=hook):
                recorder.calls.clear()
                getattr(recorder, hook)(*args, **kwargs)
                self.assertEqual(recorder.calls, [(REDEFINITIONS[hook], args, kwargs)])

    def test_default_redefinitions_are_no_ops(self):
        plain = make_callback(cls=PeriodicCallback)

        for hook, args, kwargs in HOOK_CALLS:
            with self.subTest(hook=hook):
                self.assertIsNone(getattr(plain, hook)(*args, **kwargs))

    def test_every_n_local_trainings_filters_time_based_training_hooks(self):
        hooks = (
            ('on_synchronization_start', ()),
            ('on_synchronization_end', ([],)),
            ('on_aggregation_start', ()),
            ('on_aggregation_end', ()),
            ('on_local_training_start', ()),
            ('on_local_training_end', ()),
        )
        for hook, extra in hooks:
            with self.subTest(hook=hook):
                fired = self.fired(hook, 'every_n_local_trainings', 2, [0, 1, 2, 3, 4], extra)
                self.assertEqual(fired, [0, 2, 4])

    def test_every_n_tests_filters_test_inference_hooks(self):
        for hook in ('on_test_inference_start', 'on_test_inference_end'):
            with self.subTest(hook=hook):
                fired = self.fired(hook, 'every_n_tests', 3, [0, 1, 2, 3, 4, 5, 6])
                self.assertEqual(fired, [0, 3, 6])

    def test_every_n_local_epochs_filters_epoch_hooks(self):
        for hook in ('on_training_epoch_start', 'on_training_epoch_end'):
            with self.subTest(hook=hook):
                fired = self.fired(hook, 'every_n_local_epochs', 2, [0, 1, 2, 3, 4])
                self.assertEqual(fired, [0, 2, 4])

    def test_every_n_batch_steps_filters_batch_hooks(self):
        for hook in ('on_training_batch_start', 'on_training_batch_end'):
            with self.subTest(hook=hook):
                fired = self.fired(hook, 'every_n_batch_steps', 4, [0, 1, 4, 5, 8])
                self.assertEqual(fired, [0, 4, 8])

    def test_each_period_only_affects_its_own_hooks(self):
        only_training = make_callback(every_n_local_trainings=2)

        # Odd values are skipped only by the hooks that the period governs.
        only_training.on_test_inference_end(1.0)
        only_training.on_training_epoch_end(1)
        only_training.on_training_batch_end(1)
        only_training.on_local_training_end(1.0)

        self.assertEqual([name for name, _, _ in only_training.calls],
                         ['test_inference_end', 'training_epoch_end', 'training_batch_end'])

    def test_zero_period_disables_filtering(self):
        fired = self.fired('on_local_training_start', 'every_n_local_trainings', 0, [1, 2, 3])

        self.assertEqual(fired, [1, 2, 3])

    def test_skipped_call_keeps_keyword_arguments_out_of_the_redefinition(self):
        recorder = make_callback(every_n_tests=2)

        recorder.on_test_inference_end(1.0, accuracy=0.5)
        recorder.on_test_inference_end(2.0, accuracy=0.6)

        self.assertEqual(recorder.calls, [('test_inference_end', (2.0,), {'accuracy': 0.6})])

    def test_callback_list_applies_each_callback_own_period(self):
        every_second = make_callback(every_n_local_trainings=2)
        always = make_callback()

        callbacks = CallbackList([every_second, always])
        for time in (1.0, 2.0):
            callbacks.on_local_training_end(time)

        self.assertEqual([args[0] for _, args, _ in every_second.calls], [2.0])
        self.assertEqual([args[0] for _, args, _ in always.calls], [1.0, 2.0])


class DummyCallbackTest(unittest.TestCase):
    def test_accepts_any_constructor_arguments(self):
        DummyCallback()
        DummyCallback(MagicMock(), MagicMock(), foo='bar')

    def test_every_hook_is_a_no_op(self):
        dummy = DummyCallback()

        for hook, args, kwargs in HOOK_CALLS:
            with self.subTest(hook=hook):
                self.assertIsNone(getattr(dummy, hook)(*args, **kwargs))


if __name__ == '__main__':
    unittest.main()
