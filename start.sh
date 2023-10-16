#! /bin/bash

pip install -r requirements.txt

echo -e "#! /bin/bash \necho \"app running on port 60267\"\npython3 -m shiny run --port 60267" > ./start.sh
./start.sh
