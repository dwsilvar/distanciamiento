import pandas as pd


def prepare_data(input_file):
    df = pd.read_csv(input_file, header=None)
    df.columns = ['numPersona', 'numFrame', 'headValid', 'bodyValid',
                  'headLeft', 'headTop', 'headRight', 'headBottom',
                  'bodyLeft', 'bodyTop', 'bodyRight', 'bodyBottom']

    df.drop(columns=['headValid', 'bodyValid', 'headLeft', 'headTop', 'headRight', 'headBottom'], inplace=True)

    df['bodyWidth'] = df['bodyRight'] - df['bodyLeft']
    df['posXBody'] = df[['bodyRight', 'bodyLeft']].mean(axis=1)
    df['posYBody'] = df['bodyBottom']

    return df


def process_df2bbox(data):
    index_people = (data['numPersona'].tolist())

    p_ini_bounding_box = tuple(zip(data['bodyLeft'], data['bodyTop']))
    p_end_bounding_box = tuple(zip(data['bodyRight'], data['bodyBottom']))
    pair_points_rectangle = tuple(zip(p_ini_bounding_box, p_end_bounding_box))

    x_bbox_center = data[['bodyRight', 'bodyLeft']].mean(axis=1).astype(int)
    y_bbox_center = data[['bodyBottom', 'bodyTop']].mean(axis=1).astype(int)
    p_bbox_center = tuple(zip(x_bbox_center, y_bbox_center))

    return index_people, p_bbox_center, pair_points_rectangle


def process_df2bev(data):
    index_people = (data['numPersona'].tolist())

    x_bbox_center = data[['bodyRight', 'bodyLeft']].mean(axis=1).astype(int)
    y_bb_base = data['bodyBottom'].astype(int)
    p_base_center = tuple(zip(x_bbox_center, y_bb_base))

    return index_people, p_base_center
