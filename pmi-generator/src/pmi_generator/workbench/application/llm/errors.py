class LlmRuntimeError(RuntimeError):
    pass


class TechnicalLlmError(LlmRuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class GenerationLengthError(TechnicalLlmError):
    def __init__(self) -> None:
        super().__init__(
            "Ответ модели оборван по finish_reason=length",
            retryable=False,
        )
        self.finish_reason = "length"


class ToolContractError(LlmRuntimeError):
    pass


class AttemptDiscardedError(LlmRuntimeError):
    pass
