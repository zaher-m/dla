"""Layout validation: page routing, reference assembly, deterministic checks.

Kept as its own package rather than growing `core` because validation is a
separate concern with a separate lifecycle -- it consumes normalized layout
output and the PDF, and never imports a model.
"""
