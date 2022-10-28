from fastai.vision.all import *
import gradio as gr

learn_inf = load_learner('export.pkl')

categories = ('Snake', 'Gecko')


def classify_img(img):
    pred, idx, probs = learn_inf.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['../input/gecko-snake-examples/gecko.jpg.jpg', '../input/gecko-snake-examples/snake.jpeg']

intf = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)

