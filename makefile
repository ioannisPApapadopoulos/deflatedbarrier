
compliance:
	cd examples/cantilever && python3 cantilever.py
	cd examples/mbb-beam && python3 mbb-beam.py

iterationtables:
	cd examples/double-pipe && python3 table-double-pipe.py
	cd examples/roller-pump && python3 table-roller-pump.py
	cd examples/neumann-outlet-double-pipe && python3 table-neumann-outlet-double-pipe.py

fiveholes:
	cd examples/five-holes && python3 five-holes.py
