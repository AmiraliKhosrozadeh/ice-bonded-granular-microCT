# Schritt-für-Schritt-Anleitung: Export der `ROI_Ice` aus Dragonfly

Diese Anleitung beschreibt schrittweise, wie du die Eis-ROI aus Dragonfly exportierst und die notwendigen Schritte für das Post-Processing vorbereitest.

## 1. Vorbereitung

1. Öffne Dragonfly.
2. Lade deine TIFF-Dateien als Datensatz in Dragonfly.
3. Stelle sicher, dass du den Ordner kennst, in dem die TIFF-Dateien liegen, und ein Beispiel-Dateiname.

## 2. Segmentierung vorbereiten

1. Öffne `dragonfly_scripts/scan_segmentation.py`.
2. Ändere oben im Skript diese Werte auf deine Daten:

   - `TIFF_DIR` → Ordner mit den TIFF-Dateien
   - `FILE_PREFIX` → gemeinsamer Präfix der TIFF-Dateien vor der Nummer
   - `FIRST_SLICE`, `LAST_SLICE` → erster und letzter TIFF-Slice
   - `X_MIN`, `X_MAX`, `Y_MIN`, `Y_MAX` → XY-Ausschnitt
   - `VOXEL_SIZE_UM` → falls bekannt, sonst belassen
   - `MASK_METHOD = 'cylinder'` oder `'contour'`

3. Wenn du `MASK_METHOD = 'cylinder'` nutzt, musst du in Dragonfly
   einen zylindrischen ROI erstellen und die Werte von:
   - `CYL_CAP1_UM`
   - `CYL_CAP2_UM`
   - `CYL_RADIUS_UM`
   - `CT_ORIGIN_UM`

   eintragen.

> Die wichtigsten Parameter sind `TIFF_DIR`, `FILE_PREFIX`, `FIRST_SLICE/LAST_SLICE` und die XY-Crop-Werte.

## 3. Segmentation ausführen in Dragonfly

1. Öffne die Python-Konsole in Dragonfly.
2. Führe aus:

```python
exec(open(r"C:\Users\Lennard\Micro-CT\dragonfly_scripts\scan_segmentation.py", encoding='utf-8').read(), globals())
```

3. Das Skript lädt dann die TIFFs, segmentiert und speichert das Ergebnis in `OUT_DIR`.

## 4. Eis-Maske exportieren

Nachdem `scan_segmentation.py` gelaufen ist, erzeugt es normalerweise in Dragonfly eine ROI namens `ROI_Ice`.

1. Öffne `dragonfly_scripts/dragonfly_export_ice_roi.py`.
2. Passe `OUT_TIF` an den richtigen Pfad an, z. B.:

```python
OUT_TIF = r'C:\Users\Lennard\Desktop\Data_Work_Part\alumina\Alumina_75_1000_T5\Alumina_75_1000_T5_spam\data\ice_mask_scan01.tif'
```

3. In Dragonfly-Konsole ausführen:

```python
exec(open(r"C:\Users\Lennard\Micro-CT\dragonfly_scripts\dragonfly_export_ice_roi.py", encoding='utf-8').read(), globals())
```

4. Prüfe, ob die Datei erstellt wurde.

## 5. Optional: weitere Dragonfly-Skripte

Falls du danach bead-Analyse oder Histogramm-Auswertung machen willst:

- `dragonfly_scripts/scan_histogram_analysis.py`
- `dragonfly_scripts/scan_bead_analysis.py`
- `dragonfly_scripts/scan_porosity_analysis.py`

Diese laufen ebenfalls in Dragonfly.

## 6. Post-Processing unter WSL (optional)

Wenn du danach die eigentliche Post-Processing-Pipeline starten willst, brauchst du:
- einen WSL-Workspace
- die bereits exportierten Dateien im `<spam>/data/` Ordner:
  - `ct_scan01.tif`
  - `specimen_mask_scan01.tif`
  - `bead_labels_scan01.tif`
  - `ice_mask_scan01.tif`

Dann laufen die Skripte in dieser Reihenfolge im WSL-Terminal:

```bash
python spam_<pre>_permeability.py
python spam_<pre>_crack_sparse_graph.py
python spam_<pre>_bond_sparse_graph.py
python spam_<pre>_crack_validate_with_ct.py
python spam_<pre>_ice_sparse_graph.py
python spam_<pre>_sparse_graph_compare.py
python spam_<pre>_crack_paths.py
python spam_<pre>_crack_voids.py
```

## Kurzfassung

1. `scan_segmentation.py` anpassen
2. In Dragonfly ausführen
3. `dragonfly_export_ice_roi.py` anpassen und ausführen
4. Ergebnis prüfen: segmentierte TIFFs + `ice_mask_scan01.tif`
5. Optional: WSL-Postprocessing starten


Wenn du möchtest, kann ich die Datei auch an einem anderen Ort ablegen oder eine englische Version erzeugen. Sag mir kurz, welche Variante du bevorzugst.

## Glossar — Fachbegriffe

- ROI: "Region of Interest" — ein markierter Bereich im Bild/Volumen, z. B. ein Zylinder oder eine freie Maske.
- `ROI_Ice`: Die spezielle ROI, die das Eis im CT-Volumen markiert (wird von den Skripten erstellt).
- TIFF / TIF: Standard-Bilddateiformat für einzelne Slices; ein Stapel von TIFFs bildet das 3D-Volumen.
- CT (Computertomographie): Röntgenbasierte 3D-Bildgebung; liefert Grauwertvolumina.
- Voxel: Volumenelement (3D-Pixel). Ein einzelner 3D-Bildpunkt mit einer Grauwerteigenschaft.
- Voxel-Größe / `VOXEL_SIZE_UM`: Physikalische Kantenlänge eines Voxels in Mikrometern (µm).
- Crop: Auschnitt eines Bildes/Volumens (begrenzter XY-Bereich), um nur den relevanten Bereich zu laden.
- Maske (Mask): Binäres Volumen (0/1), das Bereiche ein-/ausblendet (z. B. Probe vs. Außenluft).
- Segmentierung: Automatische Klassifikation jedes Voxels in Phasen (z. B. Luft, Eis, Glas, Aluminium).
- Otsu (Otsu-Schwelle): Automatische Methode zur Bestimmung eines Grauwerte-Schwellenwerts zur Binärtrennung.
- Konturdetektion (Contour detection): 2D-Verfahren, das die Umrisse großer zusammenhängender Objekte auf einer Scheibe findet.
- Zylinder-ROI: Gekippter/zylindrischer Region-ROI, oft per Hand in Dragonfly gemessen, dient als grobe Probenbegrenzung.
- Medianfilter: Rauschunterdrückung, ersetzt jeden Wert durch den Median im Nachbarschaftsfenster.
- Connected Component (CC): Zusammenhängende Gruppe von 'True' Pixeln in einer binären Maske.
- Watershed: 2D/3D Segmentierungs-Algorithmus, oft zur Trennung berührender Partikel verwendet.
- Bead-Labels / `bead_labels_scan{N}.tif`: 3D-Labelbild, in dem jede Kugel/Partikel eine eigene ID hat.
- SPAM: Name der Postprocessing-Pipeline/Ordnerstruktur in diesem Projekt (nicht Spam-E-Mail).
- Aligned / Ausgerichtet: Volumina, die in ein gemeinsames Weltkoordinatensystem transformiert und auf gemeinsame Form gepaddet sind.
- Sparse-Graph: Netzdarstellung (Knoten/Kanten) der Skelettstruktur einer Phase (z. B. Eis-Skelett).
- Skelett (Skeleton): Mittellinien-Repräsentation einer Phase; reduziert Volumen auf Knoten und Kanten.
- Knoten (Node) / Kante (Edge): Elemente eines Graphen; Knoten sind Verzweigungen, Kanten Verbindungen zwischen ihnen.
- DBSCAN: Dichtes-basierter Cluster-Algorithmus zur Gruppierung räumlicher Punkte (z. B. gebrochene Kanten zu Rissen).
- Permeabilität: Maß, wie leicht ein Fluid durch das poröse Material fließt; wird z. B. mit Kozeny-Carman abgeschätzt.
- Kozeny-Carman: Semi-empirische Beziehung zur Abschätzung der Permeabilität aus Porosität und spezifischer Oberfläche.
- PuMA: Dragonfly-internes Tool/Modul für Material-Metriken (optional in den Skripten aufrufbar).
- `tifffile`: Python-Bibliothek zum Lesen/Schreiben von TIFFs.
- `PIL` / `Pillow`: Alternative Python-Bibliothek für Bild-IO; Fallback, wenn `tifffile` fehlt.
- `numpy.ndarray`: Mehrdimensionales Array-Objekt in Python, das Volumina/Grauwertdaten hält.
- `uint8`, `uint16`: Datentypen (8-/16-Bit unsigned integers) für Bilddaten; Masken sind oft `uint8`, CT meist `uint16`.
- photometric / compression (in `tifffile`): Optionen zur TIFF-Speicherung (Anzeige-Interpretation, Kompressionsmethode).

Wenn du willst, kann ich jede dieser Definitionen noch ausführlicher erklären oder Beispiele/Illustrationen hinzufügen.