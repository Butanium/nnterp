class TLImportErrorProxy:
    def __init__(self, original_error):
        self.original_error = original_error

    def __call__(self, *args, **kwargs):
        print(
            f"You're trying to use UnifiedTransformer but TransformerLens is not installed. "
            "The following error was caught when importing nnterp:"
        )
        raise self.original_error

    @classmethod
    def __instancecheck__(cls, instance):
        return False


try:
    from nnsight.models.UnifiedTransformer import UnifiedTransformer
except ImportError as e:
    UnifiedTransformer = TLImportErrorProxy(e)
