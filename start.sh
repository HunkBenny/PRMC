#! /bin/bash

pip install -r requirements.txt

echo -e "#! /bin/bash \npython -m shiny run --port 60267" > ./start.sh
./start.sh
