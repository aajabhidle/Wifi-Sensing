#!/bin/bash
cat <<'TXT'
Confusion matrix (row %, leave-one-person-out, 4 persons)
True \ Predicted |    bhu |    lhu |    rhu
-----------------+--------+--------+--------
bhu              |  31.0% |  35.9% |  33.1%
lhu              |  26.3% |  40.0% |  33.7%
rhu              |  26.0% |  39.4% |  34.5%
TXT
