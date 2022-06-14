import glob
import os


MAIN_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(f"{__file__})"))) + os.path.sep
PIC_SOURCE_FOLDER = MAIN_FOLDER + "vision_source" + os.path.sep
PIC_OUTPUT_FOLDER = MAIN_FOLDER + "vision_output" + os.path.sep

# ALL_HD = glob.glob(f"{PIC_SOURCE_FOLDER}{os.path.sep}**{os.path.sep}*.jpg", recursive=True)
# ALL_HD.sort()

PICS_AUTOSTRADA = glob.glob(f"{PIC_SOURCE_FOLDER}autostrada-notsmooth**{os.path.sep}*.png",
                            recursive=True)
PICS_AUTOSTRADA.copy()

PICS_FRANKFURT = glob.glob(f"{PIC_SOURCE_FOLDER}frankfurt-munch{os.path.sep}**.png", )
PICS_FRANKFURT_TRAFFIC = glob.glob(f"{PIC_SOURCE_FOLDER}frankfurt-munch-traffic{os.path.sep}**.png", )
PICS_FRANKFURT.sort()
PICS_FRANKFURT_TRAFFIC.sort()

COLORS_PATHS = glob.glob(
        f"{PIC_SOURCE_FOLDER + 'histogram_validation'}{os.path.sep}**{os.path.sep}*.png", recursive=True
)
PICS_LINES = glob.glob(f"{PIC_SOURCE_FOLDER}lines{os.path.sep}**.png", )

os.makedirs(PIC_OUTPUT_FOLDER, exist_ok=True)


def filter_paths(pic_list_paths, filters):
    """
    Filters pictures paths from steam screenshots
    Args:
        pic_list_paths:
        filters:

    Returns:

    """
    filters.sort(key=lambda x: x[0])
    for (f1l, f1h), (f2l, f2h) in zip(filters, filters[1:]):
        if f1h >= f2l:
            print("Filter is overlaping! F1: {f1h  }, F2: {f2l} ")

    new_list = set()
    for ph in pic_list_paths:
        name = os.path.basename(ph)
        num = int(name[:-6])
        for low, high in filters:
            if low <= num <= high:
                new_list.add(ph)

    new_list = list(new_list)
    # print(f"New list size: {len(new_list)}")
    return new_list


# CABIN_PICS_1 = filter_paths(ALL_HD, [(0, 20220307164615)])
# CABIN_PICS_2 = filter_paths(ALL_HD, [(0, 20220307164615),
#                                      (20220307170754, 20220307171023),
#                                      (20220307171044, 20220307171352)]
#                             )

# print(CABIN_PICS_1)
# print(CABIN_PICS_2)

if __name__ == "__main__":
    # print(PICS_AUTOSTRADA)
    print("Normal")
    print(PICS_FRANKFURT)
    print("Traffic")
    print(PICS_FRANKFURT_TRAFFIC)
