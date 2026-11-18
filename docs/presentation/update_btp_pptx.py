import copy
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile

SRC_PPTX = "E:/percom/Codebase/percom/btp___.pptx"
BACKUP_PPTX = "E:/percom/Codebase/percom/btp____backup.pptx"
OUT_PPTX_LOCAL = "E:/percom/btp_updated.pptx"
OUT_PPTX_ORIG = "E:/percom/Codebase/percom/btp___.pptx"

# Backup original if not already backed up
if not os.path.exists(BACKUP_PPTX):
    shutil.copy2(SRC_PPTX, BACKUP_PPTX)
    print(f"Backed up original to {BACKUP_PPTX}")

NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

ET.register_namespace('p', NS_P)
ET.register_namespace('a', NS_A)
ET.register_namespace('r', NS_R)

from build_ppt_slides import build_table_slide

# Read all files from source PPTX
with zipfile.ZipFile(SRC_PPTX, 'r') as zin:
    file_dict = {name: zin.read(name) for name in zin.namelist()}

# Parse slide 19 as template for tables
slide19_xml = file_dict['ppt/slides/slide19.xml']
slide19_tree = ET.fromstring(slide19_xml)

# -----------------------------------------------------------------------------
# 1. BUILD SLIDE 20: 5-Participant Results (Outline + Joints + Speed)
# -----------------------------------------------------------------------------
s20_title = "5-Participant complete gesture results (Outline + Joints + Speed)"
# Total width available is ~10,515,240 dxa
# Columns: Participant (2,700,000), Protocol A (2,600,000), Protocol B (2,600,000), Protocol C (2,615,240)
s20_col_widths = [2700000, 2600000, 2600000, 2615240]
s20_headers = [
    "Participant",
    "Protocol A (Within-Person)",
    "Protocol B (Pooled Model)",
    "Protocol C (Leave-One-Out)"
]
s20_rows = [
    ["P1 (Priyanshu)", "92.4%", "82.4%", "35.5%"],
    ["P2 (Sanchit)",   "93.0%", "78.1%", "33.3%"],
    ["P3 (Sir)",       "82.4%", "80.1%", "32.4%"],
    ["P4 (Vansh)",     "93.6%", "83.2%", "23.4%"],
    ["P5 (Vardaan)",   "96.0%", "83.0%", "37.6%"],
    ["Overall Mean",   "91.5% +- 4.8%", "87.2% +- 1.8%", "32.4% +- 5.0%"]
]
s20_notes = [
    "Test unit: complete gestures (2,629 total gestures across 5 participants). Evaluated with 3 random seeds.",
    "Outline + joints + speed achieves 91.5% in Protocol A (within-person) and 87.2% in Protocol B (multi-person pooled).",
    "Protocol C drops to 32.4% (chance 33.3%) because body silhouettes require subject calibration on unseen users."
]

slide20_tree = build_table_slide(slide19_tree, s20_title, s20_col_widths, s20_headers, s20_rows, s20_notes)
slide20_xml = ET.tostring(slide20_tree, encoding='utf-8', xml_declaration=True)

# -----------------------------------------------------------------------------
# 2. BUILD SLIDE 21: Protocol Comparison (Calibrated vs Uncalibrated)
# -----------------------------------------------------------------------------
s21_title = "Protocol comparison: Calibrated vs Uncalibrated"
# Columns: Protocol (2,600,000), Setup (3,100,000), Outline+Joints+Speed (2,000,000), Key Takeaway (2,815,240)
s21_col_widths = [2600000, 3100000, 2000000, 2815240]
s21_headers = [
    "Protocol",
    "Evaluation Setup",
    "Outline + Joints + Spd",
    "Key Takeaway"
]
s21_rows = [
    ["Protocol A (Within-Person)", "Individually calibrated (80/20 block fold)", "91.5% +- 4.8%", "High accuracy across all 5 participants (82.4% - 96.0%)"],
    ["Protocol B (Pooled Model)", "Single model trained on 5 pooled people", "87.2% +- 1.8%", "Strong generalization when training pool includes multiple bodies"],
    ["Protocol C (Leave-One-Out)", "Zero-shot on unseen participant (uncalibrated)", "32.4% +- 5.0%", "Silhouette shift biases to BHR; requires person calibration"]
]
s21_notes = [
    "Complete gestures evaluated end-to-end with strict zero data leakage.",
    "A 1.7s RF isolation guard band drops boundary segments to eliminate CSI multipath overlap between train and test.",
    "Per-recording zero-centering and scaling strictly use training frames only."
]

slide21_tree = build_table_slide(slide19_tree, s21_title, s21_col_widths, s21_headers, s21_rows, s21_notes)
slide21_xml = ET.tostring(slide21_tree, encoding='utf-8', xml_declaration=True)

# -----------------------------------------------------------------------------
# 3. BUILD SLIDE 22: Updated Outputs & Code (Former Slide 20 with Technical Report Link)
# -----------------------------------------------------------------------------
slide20_old_xml = file_dict['ppt/slides/slide20.xml']
slide22_tree = ET.fromstring(slide20_old_xml)
ns = {'a': NS_A, 'p': NS_P, 'r': NS_R}
spTree = slide22_tree.find('.//p:spTree', ns)

# Add Text 6: Technical Report Label
sp6 = ET.SubElement(spTree, f"{{{NS_P}}}sp")
nvSpPr6 = ET.SubElement(sp6, f"{{{NS_P}}}nvSpPr")
ET.SubElement(nvSpPr6, f"{{{NS_P}}}cNvPr", {"id": "226", "name": "Text 6"})
ET.SubElement(nvSpPr6, f"{{{NS_P}}}cNvSpPr")
ET.SubElement(nvSpPr6, f"{{{NS_P}}}nvPr")

spPr6 = ET.SubElement(sp6, f"{{{NS_P}}}spPr")
xfrm6 = ET.SubElement(spPr6, f"{{{NS_A}}}xfrm")
ET.SubElement(xfrm6, f"{{{NS_A}}}off", {"x": "841320", "y": "3650000"})
ET.SubElement(xfrm6, f"{{{NS_A}}}ext", {"cx": "10514520", "cy": "420000"})
prstGeom6 = ET.SubElement(spPr6, f"{{{NS_A}}}prstGeom", {"prst": "rect"})
ET.SubElement(prstGeom6, f"{{{NS_A}}}avLst")
ET.SubElement(spPr6, f"{{{NS_A}}}noFill")
ln6 = ET.SubElement(spPr6, f"{{{NS_A}}}ln", {"w": "0"})
ET.SubElement(ln6, f"{{{NS_A}}}noFill")

txBody6 = ET.SubElement(sp6, f"{{{NS_P}}}txBody")
bodyPr6 = ET.SubElement(txBody6, f"{{{NS_A}}}bodyPr", {"lIns": "0", "rIns": "0", "tIns": "0", "bIns": "0", "anchor": "t"})
ET.SubElement(bodyPr6, f"{{{NS_A}}}noAutofit")
p6 = ET.SubElement(txBody6, f"{{{NS_A}}}p")
pPr6 = ET.SubElement(p6, f"{{{NS_A}}}pPr")
lnSpc6 = ET.SubElement(pPr6, f"{{{NS_A}}}lnSpc")
ET.SubElement(lnSpc6, f"{{{NS_A}}}spcPct", {"val": "100000"})
ET.SubElement(pPr6, f"{{{NS_A}}}buNone")

r6 = ET.SubElement(p6, f"{{{NS_A}}}r")
rPr6 = ET.SubElement(r6, f"{{{NS_A}}}rPr", {
    "b": "1", "lang": "en-US", "sz": "1800", "spc": "-1", "strike": "noStrike"
})
solidFill6 = ET.SubElement(rPr6, f"{{{NS_A}}}solidFill")
ET.SubElement(solidFill6, f"{{{NS_A}}}srgbClr", {"val": "1f1f1f"})
ET.SubElement(rPr6, f"{{{NS_A}}}latin", {"typeface": "Calibri"})
t6 = ET.SubElement(r6, f"{{{NS_A}}}t")
t6.text = "Comprehensive Technical Report & Architecture Defense:"

# Add Text 7: Technical Report Hyperlink
TECH_REPORT_URL = "https://drive.google.com/file/d/1v-TdC5vhXvgaRGnUXHkq7qXbiurIuU1Z/view?usp=drive_link"

sp7 = ET.SubElement(spTree, f"{{{NS_P}}}sp")
nvSpPr7 = ET.SubElement(sp7, f"{{{NS_P}}}nvSpPr")
cNvPr7 = ET.SubElement(nvSpPr7, f"{{{NS_P}}}cNvPr", {"id": "227", "name": "Text 7"})
ET.SubElement(cNvPr7, f"{{{NS_A}}}hlinkClick", {f"{{{NS_R}}}id": "rId5"})
ET.SubElement(nvSpPr7, f"{{{NS_P}}}cNvSpPr")
ET.SubElement(nvSpPr7, f"{{{NS_P}}}nvPr")

spPr7 = ET.SubElement(sp7, f"{{{NS_P}}}spPr")
xfrm7 = ET.SubElement(spPr7, f"{{{NS_A}}}xfrm")
ET.SubElement(xfrm7, f"{{{NS_A}}}off", {"x": "841320", "y": "4150000"})
ET.SubElement(xfrm7, f"{{{NS_A}}}ext", {"cx": "10514520", "cy": "450000"})
prstGeom7 = ET.SubElement(spPr7, f"{{{NS_A}}}prstGeom", {"prst": "rect"})
ET.SubElement(prstGeom7, f"{{{NS_A}}}avLst")
ET.SubElement(spPr7, f"{{{NS_A}}}noFill")
ln7 = ET.SubElement(spPr7, f"{{{NS_A}}}ln", {"w": "0"})
ET.SubElement(ln7, f"{{{NS_A}}}noFill")

txBody7 = ET.SubElement(sp7, f"{{{NS_P}}}txBody")
bodyPr7 = ET.SubElement(txBody7, f"{{{NS_A}}}bodyPr", {"lIns": "0", "rIns": "0", "tIns": "0", "bIns": "0", "anchor": "t"})
ET.SubElement(bodyPr7, f"{{{NS_A}}}noAutofit")
p7 = ET.SubElement(txBody7, f"{{{NS_A}}}p")
pPr7 = ET.SubElement(p7, f"{{{NS_A}}}pPr")
lnSpc7 = ET.SubElement(pPr7, f"{{{NS_A}}}lnSpc")
ET.SubElement(lnSpc7, f"{{{NS_A}}}spcPct", {"val": "100000"})
ET.SubElement(pPr7, f"{{{NS_A}}}buNone")

r7 = ET.SubElement(p7, f"{{{NS_A}}}r")
rPr7 = ET.SubElement(r7, f"{{{NS_A}}}rPr", {
    "b": "1", "lang": "en-US", "sz": "1700", "spc": "-1", "strike": "noStrike", "u": "sng"
})
solidFill7 = ET.SubElement(rPr7, f"{{{NS_A}}}solidFill")
ET.SubElement(solidFill7, f"{{{NS_A}}}srgbClr", {"val": "0563c1"})
ET.SubElement(rPr7, f"{{{NS_A}}}latin", {"typeface": "Calibri"})
ET.SubElement(rPr7, f"{{{NS_A}}}hlinkClick", {f"{{{NS_R}}}id": "rId6"})
t7 = ET.SubElement(r7, f"{{{NS_A}}}t")
t7.text = TECH_REPORT_URL

slide22_xml = ET.tostring(slide22_tree, encoding='utf-8', xml_declaration=True)

# Build slide22 rels (copying slide20 rels and adding rId5, rId6)
slide20_old_rels = file_dict['ppt/slides/_rels/slide20.xml.rels'].decode('utf-8')
# Insert relationship for rId5 and rId6
rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
tree_rels = ET.fromstring(slide20_old_rels)
ET.SubElement(tree_rels, f"{{{rel_ns}}}Relationship", {
    "Id": "rId5",
    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
    "Target": TECH_REPORT_URL,
    "TargetMode": "External"
})
ET.SubElement(tree_rels, f"{{{rel_ns}}}Relationship", {
    "Id": "rId6",
    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
    "Target": TECH_REPORT_URL,
    "TargetMode": "External"
})
slide22_rels = ET.tostring(tree_rels, encoding='utf-8', xml_declaration=True)

# Slide 20 and Slide 21 rels (same as slide19 rels)
slide_table_rels = file_dict['ppt/slides/_rels/slide19.xml.rels']

# -----------------------------------------------------------------------------
# 4. UPDATE PRESENTATION.XML & PRESENTATION.XML.RELS
# -----------------------------------------------------------------------------
# In presentation.xml, slides are listed in sldIdLst.
# We have 22 slides total now.
pres_tree = ET.fromstring(file_dict['ppt/presentation.xml'])
sldIdLst = pres_tree.find('p:sldIdLst', {'p': NS_P})

# Current last slide in sldIdLst corresponds to old slide 20
# Let's see: rId23 was slide20.xml
# We add rId24 (for slide21.xml) and rId25 (for slide22.xml)
# The order in sldIdLst:
# rId4 .. rId22 (slides 1 to 19)
# rId23 (slide 20: 5-participant results)
# rId24 (slide 21: protocol comparison)
# rId25 (slide 22: outputs and code)

# Let's inspect sldId elements
existing_sldIds = list(sldIdLst)
max_id = max(int(elem.attrib.get('id', 256)) for elem in existing_sldIds)

# Add slide 21 and slide 22
ET.SubElement(sldIdLst, f"{{{NS_P}}}sldId", {"id": str(max_id + 1), f"{{{NS_R}}}id": "rId24"})
ET.SubElement(sldIdLst, f"{{{NS_P}}}sldId", {"id": str(max_id + 2), f"{{{NS_R}}}id": "rId25"})
updated_pres_xml = ET.tostring(pres_tree, encoding='utf-8', xml_declaration=True)

# Update presentation.xml.rels
pres_rels_tree = ET.fromstring(file_dict['ppt/_rels/presentation.xml.rels'])
# rId23 -> slides/slide20.xml (already points to slide20.xml)
# Add rId24 -> slides/slide21.xml
# Add rId25 -> slides/slide22.xml
ET.SubElement(pres_rels_tree, f"{{{rel_ns}}}Relationship", {
    "Id": "rId24",
    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
    "Target": "slides/slide21.xml"
})
ET.SubElement(pres_rels_tree, f"{{{rel_ns}}}Relationship", {
    "Id": "rId25",
    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
    "Target": "slides/slide22.xml"
})
updated_pres_rels = ET.tostring(pres_rels_tree, encoding='utf-8', xml_declaration=True)

# Update [Content_Types].xml
ct_tree = ET.fromstring(file_dict['[Content_Types].xml'])
ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.SubElement(ct_tree, f"{{{ct_ns}}}Override", {
    "PartName": "/ppt/slides/slide21.xml",
    "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
})
ET.SubElement(ct_tree, f"{{{ct_ns}}}Override", {
    "PartName": "/ppt/slides/slide22.xml",
    "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
})
updated_ct_xml = ET.tostring(ct_tree, encoding='utf-8', xml_declaration=True)

# -----------------------------------------------------------------------------
# 5. ASSEMBLE OUTPUT PPTX
# -----------------------------------------------------------------------------
# Dictionary of final files:
final_files = dict(file_dict)
# Keep slide 1..19 exactly identical
# Slide 20 is new 5-participant results
final_files['ppt/slides/slide20.xml'] = slide20_xml
final_files['ppt/slides/_rels/slide20.xml.rels'] = slide_table_rels

# Slide 21 is new comparison slide
final_files['ppt/slides/slide21.xml'] = slide21_xml
final_files['ppt/slides/_rels/slide21.xml.rels'] = slide_table_rels

# Slide 22 is updated outputs and code
final_files['ppt/slides/slide22.xml'] = slide22_xml
final_files['ppt/slides/_rels/slide22.xml.rels'] = slide22_rels

# Structure files
final_files['ppt/presentation.xml'] = updated_pres_xml
final_files['ppt/_rels/presentation.xml.rels'] = updated_pres_rels
final_files['[Content_Types].xml'] = updated_ct_xml

# Write to both OUT_PPTX_LOCAL and OUT_PPTX_ORIG
for out_path in [OUT_PPTX_LOCAL, OUT_PPTX_ORIG]:
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for fname, data in final_files.items():
            zout.writestr(fname, data)
    print(f"Successfully generated updated PPTX: {out_path} ({os.path.getsize(out_path)} bytes)")

print("\nAll done!")
