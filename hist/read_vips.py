
def parse_vips(description):
    """Return metadata from vips image description as dict."""
    if not description.startswith('<?xml version="1.0"?>'):
      raise ValueError('invalid MetaSeries image description')

    import xml.etree.ElementTree as etree
    import re

    root = etree.fromstring(description)
    #ns = re.match('\{(.*)\}', root.tag).group(1)
    ns = re.match('\{(.*)\}', root.tag).group(0)
    
    #image =  root.find('image')
    #etree.dump(image)
    #print(image)
    #quit()

    types = {
        'float': float,
        'int': int,
        'bool': lambda x: asbool(x, 'on', 'off'),
        'VipsRefString' : str,
        'gint' : int,
        'gdouble' : float

    }

    # def is_convertible_to_float(value):
    #   try:
    #     float(value)
    #     return True
    #   except:
    #     return False

    def parse(root, result):
      for image in root.findall(f'{ns}properties'):
        for prop in image.findall(f'{ns}property'):
          value = prop.find(f"{ns}value")
          result[prop.find(f"{ns}name").text] = types[value.get("type")](value.text)
  
      return result

    adict = parse(root, {})
    if 'Description' in adict:
        adict['Description'] = adict['Description'].replace('&#13;&#10;', '\n')
    return adict