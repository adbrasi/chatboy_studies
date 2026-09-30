#!/bin/bash
# Downloads all corpora into data/raw/. Berntzen FoLiA files are streamed and reduced to message-level TSV.
set -e
cd "$(dirname "$0")/../data" && mkdir -p raw && cd raw
mkdir -p maichat nps nus ed berntzen
curl -sSL -o maichat.zip "https://datashare.ed.ac.uk/download/DS_10283_9163.zip" && unzip -oq maichat.zip -d maichat && (cd maichat && tar xzf maichat.tar.gz)
curl -sSL -o nps/nps_chat.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/nps_chat.zip && (cd nps && unzip -oq nps_chat.zip)
curl -sSL -o nus/en_xml.zip https://raw.githubusercontent.com/kite1988/nus-sms-corpus/master/smsCorpus_en_xml_2015.03.09_all.zip && (cd nus && unzip -oq en_xml.zip)
curl -sSL -o ed/ed.tar.gz https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz && (cd ed && tar xzf ed.tar.gz)
cd berntzen
curl -sSL "https://ssh.datastations.nl/api/datasets/:persistentId/?persistentId=doi:10.17026/DANS-XZZ-UGTW" | python3 -c "
import json,sys
for f in json.load(sys.stdin)['data']['latestVersion']['files']:
  if f['label'].endswith('.folia.xml'): print(f['dataFile']['id'], f['label'])
" | while read id name; do
  out="${name%.folia.xml}.events.tsv"; [ -s "$out" ] && continue
  curl -sSL "https://ssh.datastations.nl/api/access/datafile/$id" | python3 "$(dirname "$0")/../../../scripts/folia_events.py" > "$out"
done
