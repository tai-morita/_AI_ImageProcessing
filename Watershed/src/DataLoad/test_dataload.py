import requests
import numpy as np
s = requests.Session()

account = "tai-morita"
BASE_DIR = r"D:\_study\ImageProcessing\ForNIshimura\SaveAnnotation\certs-tai-morita"

url_base = "https://192.168.17.106:50050/api"
catalogue_key = "3af616826f78654f5d6755e047df971cf296bba908e5e668699a9e3029b6f33b"
#catalogue_key = "1c74ad395cd3e226636dac37aa7d5c4c9552cc35f14533cc39c9c9111e599b0a"

catalogue_tags = ["segmentation", "test", "phase1", "上顎洞", "DentalSegmentator", "label"]
##catalogue_tags =   ["segmentation", "training", "phase1", "オトガイ孔", "DentalSegmentator", "label"]

select_index = 0

from pathlib import Path


ca_path     =  f"{BASE_DIR}\\ca.crt"
client_crt  = f"{BASE_DIR}\\{account}.client.crt"
client_key  = f"{BASE_DIR}\\{account}.client.key"
 
def loadVolumeFromRaw(content):
    import xml.etree.ElementTree as ET

    lenOfXML = int.from_bytes(content[21:23], "little")
    volxml = content[25:lenOfXML+25].decode("ASCII")
    volxml = ET.fromstring(volxml)

    xmin = int(volxml.find("FYI/XMin").attrib["value"])
    xmax = int(volxml.find("FYI/XMax").attrib["value"])
    ymin = int(volxml.find("FYI/YMin").attrib["value"])
    ymax = int(volxml.find("FYI/YMax").attrib["value"])
    zmin = int(volxml.find("FYI/ZMin").attrib["value"])
    zmax = int(volxml.find("FYI/ZMax").attrib["value"])
    xmin, xmax, ymin, ymax, zmin, zmax
    xlen=xmax-xmin+1
    ylen=ymax-ymin+1
    zlen=zmax-zmin+1

    bIsHuAvailableElem = volxml.find("Attribute/bIsHuAvailable")
    bIsHuAvailable = False
    if bIsHuAvailableElem is not None:
        bIsHuAvailable = int(bIsHuAvailableElem.attrib["value"]) == 1
    else:
        print("DEBUG: bIsHuAvailable is not found.")

    if (bIsHuAvailable):
        huInter = int(volxml.find("Attribute/tfSystemV2HuIntercept").attrib["value"])
        huSlope = int(volxml.find("Attribute/tfSystemV2HuSlope").attrib["value"])
    else:
        huInter = 0
        huSlope = 100

    pxSize = float(volxml.find("Attribute/tfXGridSize").attrib["value"])

    """
    antiAliasElem = volxml.find("Attribute/tfAntiAliasAngleInDegree")
    antiAlias = 0
    if antiAliasElem is not None:
        antiAlias = float(antiAliasElem.attrib["value"])
    else:
        print("DEBUG: tfAntiAliasAngleInRadian is not found.")
        antiAlias = float(volxml.find("Attribute/tfAntiAliasAngleInRadian").attrib["value"])/np.pi*180
    """
    rotAngles=[]
    rotAngles.append(float(volxml.find("Attribute/tfAntiAliasAngleInRadian").attrib["value"])/np.pi*180+float(volxml.find("Attribute/tfInitialAngleInRadian").attrib["value"])/np.pi*180)
    rotAngles.append(float(volxml.find("Attribute/tfAntiAliasAngleInDegree").attrib["value"])+90)

    tA = float(volxml.find("Attribute/tfA").attrib["value"]);
    tB = float(volxml.find("Attribute/tfB").attrib["value"]);

    totalLen = xlen*ylen*zlen
    vol = np.frombuffer(content[lenOfXML+61:lenOfXML+61+totalLen*2], dtype=np.int16)

    vol = vol.reshape((xlen, ylen, zlen))

    vol = vol.transpose((2, 1, 0))
    vol = vol[:,:,:]
    vol = ((np.array(vol, dtype=np.float32)*tA + tB)*huSlope+huInter)

    fovArea = (vol != vol[0, 0, 0])

    vol[vol == vol[0, 0, 0]] = 0 

    return vol, pxSize, np.array([xmin, ymin, zmin])*pxSize, rotAngles, fovArea

print(f"find catalogue by {catalogue_key}...")

print("try............")
res = s.get(f"{url_base}/catalogue/{catalogue_key}", 
    cert=(client_crt, client_key),
    verify=ca_path,
)

print(res.status_code)

datasets = res.json()["nodes"]
print(f"find data num: {len(datasets)}")
print("please wait for a while.......")


res = s.get(f"{url_base}/node/{datasets[select_index]}",
    cert=(client_crt, client_key),
    verify=ca_path,
)
uid = res.json()["attachment"]

url = f"{url_base}/attach/{uid}"

res = s.get(f"{url_base}/attach/{uid}",
    cert=(client_crt, client_key),
    verify=ca_path,
)

print("Download end...")
vol, pxSize, firstPos, rotAngles, fovArea = loadVolumeFromRaw(res.content)
print("FP", firstPos)
# %matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(vol[163], cmap="gray")