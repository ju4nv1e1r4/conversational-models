# We can use this module to stablish connection or client btw

import os
from falkordb import FalkorDB

class Connections:
    def __init__(self, falkordb_client = None):
        self.falkordb_client = falkordb_client

    def our_falkordb_client(self):
        self.falkordb_client = FalkorDB(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=6379,
            password=os.getenv("FALKORDB_PASSWORD")
        )

        return self.falkordb_client
