import glob
import os.path
import numpy as np
import os

# 图片数据文件夹
INPUT_DATA = r'/mnt/Data8T/Datasets/VGG-Face2/data/test_align_crop_224_frontal'


def create_image_lists():
    matched_result = set()
    k = 0
    # 获取当前目录下所有的子目录,这里x 是一个三元组(root,dirs,files)，第一个元素表示INPUT_DATA当前目录，
    # 第二个元素表示当前目录下的所有子目录,第三个元素表示当前目录下的所有的文件
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    while len(matched_result) < 3000:
        print(len(matched_result))
        for sub_dir in sub_dirs[1:]:
            # 获取当前目录下所有的有效图片文件
            extensions = 'jpg'
            # 把图片存放在file_list列表里
            file_list = []
            # os.path.basename(sub_dir)返回sub_sir最后的文件名

            dir_name = os.path.basename(sub_dir)
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extensions)
            # glob.glob(file_glob)获取指定目录下的所有图片，存放在file_list中
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue
            # 通过目录名获取类别的名称
            label_name = dir_name
            length = len(file_list)
            random_number1 = np.random.randint(50)
            random_number2 = np.random.randint(50)
            base_name1 = os.path.basename(file_list[random_number1 % length])  # 获取文件的名称
            base_name2 = os.path.basename(file_list[random_number2 % length])

            if (file_list[random_number1 % length] != file_list[random_number2 % length]):
                # 将当前类别的数据放入结果字典
                matched_result.add(label_name + '\t' + base_name1 + '\t' + base_name2)
                k = k + 1

    # 返回整理好的所有数据
    return matched_result, k


def create_pairs():
    unmatched_result = set()  # 不同类的匹配对
    k = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # sub_dirs[0]表示当前文件夹本身的地址，不予考虑，只考虑他的子目录
    for sub_dir in sub_dirs[1:]:
        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg']
        file_list = []
        # 把图片存放在file_list列表里

        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # glob.glob(file_glob)获取指定目录下的所有图片，存放在file_list中
            file_list.extend(glob.glob(file_glob))

    length_of_dir = len(sub_dirs)
    for j in range(24):
        for i in range(length_of_dir):
            class1 = sub_dirs[i]
            class2 = sub_dirs[(length_of_dir - i + j - 1) % length_of_dir]
            if ((length_of_dir - i + j - 1) % length_of_dir):
                class1_name = os.path.basename(class1)
                class2_name = os.path.basename(class2)
                # 获取当前目录下所有的有效图片文件
                extension = 'jpg'
                file_list1 = []
                file_list2 = []
                # 把图片存放在file_list列表里
                file_glob1 = os.path.join(INPUT_DATA, class1_name, '*.' + extension)
                file_list1.extend(glob.glob(file_glob1))
                file_glob2 = os.path.join(INPUT_DATA, class2_name, '*.' + extension)
                file_list2.extend(glob.glob(file_glob2))
                if file_list1 and file_list2:
                    base_name1 = os.path.basename(file_list1[j % len(file_list1)])  # 获取文件的名称
                    base_name2 = os.path.basename(file_list2[j % len(file_list2)])
                    # unmatched_result.add([class1_name, base_name1, class2_name, base_name2])
                    s = class2_name + '\t' + base_name2 + '\t' + class1_name + '\t' + base_name1
                    if s not in unmatched_result:
                        unmatched_result.add(class1_name + '\t' + base_name1 + '\t' + class2_name + '\t' + base_name2)
                    k = k + 1
    return unmatched_result, k


result, k1 = create_image_lists()
print(len(result))
# print(result)

result_un, k2 = create_pairs()
print(len(result_un))
# print(result_un)

file = open('pairs_vggface2.txt', 'w')

result1 = list(result)
result2 = list(result_un)

file.write('10 300\n')

j = 0
for i in range(10):
    j = 0
    print("=============================================第" + str(i) + '次, 相同的')
    for pair in result1[i * 300:i * 300 + 300]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')

    print("=============================================第" + str(i) + '次, 不同的')
    for pair in result2[i * 300:i * 300 + 300]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')

file.close()