#! /bin/bash

pip install -r requirements.txt

echo -e "#! /bin/bash\nopen http://localhost:60267 \npython -m shiny run --port 60267" > ./start.sh
./start.sh
