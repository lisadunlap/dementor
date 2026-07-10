"""``python -m dementor.training.matrix`` entrypoint (delegates to ``main``)."""
import sys

from . import main

sys.exit(main())
