from PIL import Image, ImageDraw
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

ofa_pipe = pipeline(
    Tasks.visual_grounding,
    model='damo/ofa_visual-grounding_refcoco_large_zh'
)
def visual_grouding_do(num, text):
    image_path = './static/img/'+num+'.jpg'
    input = {'image': image_path, 'text': text}
    result = ofa_pipe(input)
    image = Image.open(image_path)
    boxes = result[OutputKeys.BOXES]
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=4)
    output_image_path = './static/result/'+num+'.jpg'
    image.save(output_image_path)


