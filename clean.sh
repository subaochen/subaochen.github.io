#!/bin/bash

# delete lyx/dia/'s backup file
find ./ -name "*.*~" -exec rm {} \;
find ./ -name "*474.lyx" -exec rm {} \;
find ./ -name "#*.*#" -exec rm {} \;
