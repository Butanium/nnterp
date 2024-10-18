class TLImportError:
    def __init__(self, e):
        self.e = e

    def __call__(self, *args, **kwargs):
        print(
            "You're trying to use TransformerLens but you don't have it installed. The following error was caught when you imported nnterp:"
        )
        raise self.e


try:
    from nnsight.models.UnifiedTransformer import UnifiedTransformer
except ImportError as e:
    UnifiedTransformer = TLImportError(e)
