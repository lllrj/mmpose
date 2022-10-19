dataset_info = dict(
    dataset_name='Air',
    paper_info=dict(
        author='digger'
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Airport: syn and real',
        container='European conference on computer vision',
        year='2022',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_inner_wing',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_inner_wing'),
        2:
        dict(
            name='left_out_wing',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right_out_wing'),
        3:
        dict(
            name='left_horizontal_inner_wing',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_horizontal_inner_wing'),
        4:
        dict(
            name='left_horizontal_out_wing',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='right_horizontal_out_wing'),
        5:
        dict(
            name='right_inner_wing',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='left_inner_wing'),
        6:
        dict(
            name='right_out_wing',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_out_wing'),
        7:
        dict(
            name='right_horizontal_inner_wing',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='left_horizontal_inner_wing'),
        8:
        dict(
            name='right_horizontal_out_wing',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_horizontal_out_wing'),
        9:
        dict(
            name='lower_vertical_wing',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        10:
        dict(
            name='upper_vertical_wing',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        11:
        dict(
            name='fuselage',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('head', 'fuselage'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_inner_wing', 'fuselage'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_inner_wing', 'fuselage'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('left_inner_wing', 'left_out_wing'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_inner_wing', 'right_out_wing'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_horizontal_inner_wing', 'left_horizontal_out_wing'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_horizontal_inner_wing', 'right_horizontal_out_wing'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('lower_vertical_wing', 'upper_vertical_wing'),id=7,color=[51, 153, 255]),
        8:
        dict(link=('left_horizontal_inner_wing', 'fuselage'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('right_horizontal_inner_wing', 'fuselage'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_horizontal_inner_wing', 'upper_vertical_wing'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_horizontal_inner_wing', 'upper_vertical_wing'), id=11, color=[255, 128, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.072, 0.079, 0.062, 0.072, 0.062, 0.079, 0.062, 0.072, 0.062, 0.026, 0.025, 0.107
    ])
