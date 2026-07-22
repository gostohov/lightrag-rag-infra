CARD_FIELD_PATHS = frozenset(
    {
        "requirement.condition",
        "requirement.preconditions",
        "requirement.behavior",
        "requirement.consequences",
        "test.initial_state",
        "test.preconditions",
        "test.previous_commands",
        "test.action",
        "test.changed_factor",
        "test.control_values",
        "test.command.cla",
        "test.command.ins",
        "test.command.p1",
        "test.command.p2",
        "test.command.lc",
        "test.command.data",
        "test.command.le",
        "test.expected.status_word",
        "test.expected.response_data",
        "test.expected.state_change",
        "test.expected.no_state_change",
        "test.observation.kind",
        "test.observation.method",
        "test.observation.value",
        "test.observation.causal_link",
        "test.observation.alternative_causes",
        "test.observation.exclusions",
    }
)


REQUIRED_FIELD_PATHS = frozenset(
    {
        "requirement.condition",
        "requirement.behavior",
        "test.action",
        "test.changed_factor",
        "test.observation.method",
    }
)


EXPECTED_RESULT_PATHS = frozenset(
    {
        "test.expected.status_word",
        "test.expected.response_data",
        "test.expected.state_change",
        "test.expected.no_state_change",
    }
)
