def tl_import_error(e):
    def raise_error(*args, **kwargs):
        print(
            "You're trying to use TransformerLens but you don't have it installed. The following error was catched when you imported nnterp:"
        )
        raise e

    return raise_error


try:
    from nnsight.models.UnifiedTransformer import UnifiedTransformer
except ImportError as e:
    UnifiedTransformer = tl_import_error(e)
