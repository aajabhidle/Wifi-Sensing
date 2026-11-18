import zipfile
import xml.etree.ElementTree as ET

with zipfile.ZipFile('E:/percom/Codebase/percom/btp___.pptx', 'r') as z:
    tree = ET.fromstring(z.read('ppt/slides/slide19.xml'))
    ns = {
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
    }
    for idx, sp in enumerate(tree.findall('.//p:sp', ns)):
        nv = sp.find('.//p:cNvPr', ns)
        xfrm = sp.find('.//a:xfrm', ns)
        x = xfrm.find('a:off', ns).attrib['x'] if xfrm is not None else ''
        y = xfrm.find('a:off', ns).attrib['y'] if xfrm is not None else ''
        cx = xfrm.find('a:ext', ns).attrib['cx'] if xfrm is not None else ''
        cy = xfrm.find('a:ext', ns).attrib['cy'] if xfrm is not None else ''
        nid = nv.attrib.get('id') if nv is not None else ''
        name = nv.attrib.get('name') if nv is not None else ''
        print(f"Shape {idx}: id={nid} name={name} x={x} y={y} cx={cx} cy={cy}")

    gf = tree.find('.//p:graphicFrame', ns)
    if gf is not None:
        nv = gf.find('.//p:cNvPr', ns)
        xfrm = gf.find('.//p:xfrm', ns)
        x = xfrm.find('a:off', ns).attrib['x'] if xfrm is not None else ''
        y = xfrm.find('a:off', ns).attrib['y'] if xfrm is not None else ''
        cx = xfrm.find('a:ext', ns).attrib['cx'] if xfrm is not None else ''
        cy = xfrm.find('a:ext', ns).attrib['cy'] if xfrm is not None else ''
        nid = nv.attrib.get('id') if nv is not None else ''
        name = nv.attrib.get('name') if nv is not None else ''
        print(f"Table: id={nid} name={name} x={x} y={y} cx={cx} cy={cy}")
