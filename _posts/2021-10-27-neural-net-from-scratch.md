# Building a neural network from scratch



I started learning about programming four years ago and one of the motivations was that I wanted to understand how "Artificial Intelligence" (AI) actually worked. I saw how computers were able to recognize objects and that seemed pretty magical to me. (more specific example: how Tesla's can actually "see" the world with their onboard cameras and navigate in it)

Yesterday was quite a breakthrough for me because after all those years I came to a point where I had enough programming skills to actually code a neural net from scratch (following Fast AI's course) where I understand all the actual steps involved, and where the multiplication and addition of a lot of numbers resulted in the computer being able to predict handwritten digits. I have to say: it's quite a magical experience where you see the model starting to get more and more accurate. Just by doing multiplications and additions. Pure math...

I\'m currently going through the fast.ai course called "Practical Deep Learning for Coders" and in lesson 4 we trained a basic neural net where it predicts if a handwritten digit is a 3 or a 7. Using that knowledge students are then encouraged to take it a step further and make it pick between all the digits from 0 to 9.

In order to train a neural net you need a lof of images to train the model on: MNIST is one of the most famous datasets in computer vision and was used by Yan Lecun in 1998 to produce the first useful model for handwritten digit recognition. This was one of the major breakthoughs for AI back in the day.
The dataset contains 60,000 images of 28 by 28 pixels of different digits (0 to 9) written by hand. The data is divided in two completely separate sets. One for "training" the model and one for validating so that we can check the accuracy or how well the model predicts digits. That way we can see the result of our training and hopefully the accuracy of the model becomes better.

In this blog post I will run through all the steps I took to solve this problem. A lot of inspiration obviously came from the fast.ai course but a lot I had to research myself in the PyTorch documentation, fast.ai forum and of course Google searches.

This walk-through assumes basic programming knowledge (Python) and a grasp of the basics of machine learning and neural networks, but I'll try to make it as easy as possible to understand without in-depth ML knowledge.

## Dependencies

Let\'s get started by installing the required dependencies. Using fastai also imports all the required pytorch libraries that are needed.

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')
```

Using the fastai untar_data() method, we can very easily download and extract the full MNIST dataset

```python
path = untar_data(URLs.MNIST)
```

```python
# sets the path of the database as the base path, making sure we use relative paths later on, making it easier to read
Path.BASE_PATH = path
```

```python
# ls() method shows there are two folders in the dataset.
path.ls()
```




    (#2) [Path('training'),Path('testing')]



```python
# inside the training folder we have new folders for all digits
(path/'training').ls()
```




    (#10) [Path('training/9'),Path('training/0'),Path('training/7'),Path('training/6'),Path('training/1'),Path('training/8'),Path('training/4'),Path('training/3'),Path('training/2'),Path('training/5')]



Now that we know where the actual images of the digits are stored, we can use all the paths to the images and store them in variables like below

```python
zeros = (path/'training'/'0').ls().sorted()
ones = (path/'training'/'1').ls().sorted()
twos = (path/'training'/'2').ls().sorted()
threes = (path/'training'/'3').ls().sorted()
fours = (path/'training'/'4').ls().sorted()
fives = (path/'training'/'5').ls().sorted()
sixes = (path/'training'/'6').ls().sorted()
sevens = (path/'training'/'7').ls().sorted()
eights = (path/'training'/'8').ls().sorted()
nines = (path/'training'/'9').ls().sorted()
```

Let's see what we stored into this variable. It's a list with all the links (paths) to all the "one" digits.

```python
ones
```




    (#6742) [Path('training/1/10006.png'),Path('training/1/10007.png'),Path('training/1/1002.png'),Path('training/1/10020.png'),Path('training/1/10027.png'),Path('training/1/1003.png'),Path('training/1/10040.png'),Path('training/1/10048.png'),Path('training/1/10058.png'),Path('training/1/10067.png')...]



Now let's open the first item of that list. We store the first item (0-th item) of that list in zero_path and we then open it with the Python Image.open() method. Here we see the picture of a zero. (and a two below that)

```python
zero_path = zeros[0]
zero = Image.open(zero_path)
zero
```




![png](2021-10-27-neural-net-from-scratch_files/output_14_0.png)



```python
two_path = twos[0]
two = Image.open(two_path)
two
```




![png](2021-10-27-neural-net-from-scratch_files/output_15_0.png)



Now we will convert the list of paths, to a list of tensors using the PyTorch tensor() method and using list comprehension, which is a very intuitive way to create lists from scratch in Python. ([Click here](https://www.w3schools.com/python/python_lists_comprehension.asp) to learn more)

Tensors are in fact matrices that consist of numbers and the have dimensions. Our tensors will have a 28 by 28 dimension because that is how many pixels they have. Each pixel is represented by a number between 0 and 255 depending on the intensity.

```python
zeros_tensors = [tensor(Image.open(o)) for o in zeros]
ones_tensors = [tensor(Image.open(o)) for o in ones]
twos_tensors = [tensor(Image.open(o)) for o in twos]
threes_tensors = [tensor(Image.open(o)) for o in threes]
fours_tensors = [tensor(Image.open(o)) for o in fours]
fives_tensors = [tensor(Image.open(o)) for o in fives]
sixes_tensors = [tensor(Image.open(o)) for o in sixes]
sevens_tensors = [tensor(Image.open(o)) for o in sevens]
eights_tensors = [tensor(Image.open(o)) for o in eights]
nines_tensors = [tensor(Image.open(o)) for o in nines]
```

When we check out the length of two of these lists, we see that for example we have 5923 images of zero's, and 5958 images of two's.

```python
len(zeros_tensors), len(twos_tensors)
```




    (5923, 5958)



Let's grab the first five of the fives_tensors list and see what is looks like. It shows a tensor with dimension 28 by 28. (28 rows by 28 columns). When you look carefully at the numbers you see zero's where there is whitespace and other numbers where there is something written, hence slowly revealing a number 5

```python
fives_tensors[0]
```




    tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253, 205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,  90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253, 190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190, 253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35, 241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39, 148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221, 253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253, 253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253, 195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,  11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
            [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]], dtype=torch.uint8)



Let's now create a Pandas (another library) dataframe and add some color to it. You can really see the shape of the image getting revealed.

```python
pd.DataFrame(fives_tensors[0]).style.set_properties(**{'font-size':'4pt'}).background_gradient('Greens')
```




<style  type="text/css" >
#T_aab55_row0_col0,#T_aab55_row0_col1,#T_aab55_row0_col2,#T_aab55_row0_col3,#T_aab55_row0_col4,#T_aab55_row0_col5,#T_aab55_row0_col6,#T_aab55_row0_col7,#T_aab55_row0_col8,#T_aab55_row0_col9,#T_aab55_row0_col10,#T_aab55_row0_col11,#T_aab55_row0_col12,#T_aab55_row0_col13,#T_aab55_row0_col14,#T_aab55_row0_col15,#T_aab55_row0_col16,#T_aab55_row0_col17,#T_aab55_row0_col18,#T_aab55_row0_col19,#T_aab55_row0_col20,#T_aab55_row0_col21,#T_aab55_row0_col22,#T_aab55_row0_col23,#T_aab55_row0_col24,#T_aab55_row0_col25,#T_aab55_row0_col26,#T_aab55_row0_col27,#T_aab55_row1_col0,#T_aab55_row1_col1,#T_aab55_row1_col2,#T_aab55_row1_col3,#T_aab55_row1_col4,#T_aab55_row1_col5,#T_aab55_row1_col6,#T_aab55_row1_col7,#T_aab55_row1_col8,#T_aab55_row1_col9,#T_aab55_row1_col10,#T_aab55_row1_col11,#T_aab55_row1_col12,#T_aab55_row1_col13,#T_aab55_row1_col14,#T_aab55_row1_col15,#T_aab55_row1_col16,#T_aab55_row1_col17,#T_aab55_row1_col18,#T_aab55_row1_col19,#T_aab55_row1_col20,#T_aab55_row1_col21,#T_aab55_row1_col22,#T_aab55_row1_col23,#T_aab55_row1_col24,#T_aab55_row1_col25,#T_aab55_row1_col26,#T_aab55_row1_col27,#T_aab55_row2_col0,#T_aab55_row2_col1,#T_aab55_row2_col2,#T_aab55_row2_col3,#T_aab55_row2_col4,#T_aab55_row2_col5,#T_aab55_row2_col6,#T_aab55_row2_col7,#T_aab55_row2_col8,#T_aab55_row2_col9,#T_aab55_row2_col10,#T_aab55_row2_col11,#T_aab55_row2_col12,#T_aab55_row2_col13,#T_aab55_row2_col14,#T_aab55_row2_col15,#T_aab55_row2_col16,#T_aab55_row2_col17,#T_aab55_row2_col18,#T_aab55_row2_col19,#T_aab55_row2_col20,#T_aab55_row2_col21,#T_aab55_row2_col22,#T_aab55_row2_col23,#T_aab55_row2_col24,#T_aab55_row2_col25,#T_aab55_row2_col26,#T_aab55_row2_col27,#T_aab55_row3_col0,#T_aab55_row3_col1,#T_aab55_row3_col2,#T_aab55_row3_col3,#T_aab55_row3_col4,#T_aab55_row3_col5,#T_aab55_row3_col6,#T_aab55_row3_col7,#T_aab55_row3_col8,#T_aab55_row3_col9,#T_aab55_row3_col10,#T_aab55_row3_col11,#T_aab55_row3_col12,#T_aab55_row3_col13,#T_aab55_row3_col14,#T_aab55_row3_col15,#T_aab55_row3_col16,#T_aab55_row3_col17,#T_aab55_row3_col18,#T_aab55_row3_col19,#T_aab55_row3_col20,#T_aab55_row3_col21,#T_aab55_row3_col22,#T_aab55_row3_col23,#T_aab55_row3_col24,#T_aab55_row3_col25,#T_aab55_row3_col26,#T_aab55_row3_col27,#T_aab55_row4_col0,#T_aab55_row4_col1,#T_aab55_row4_col2,#T_aab55_row4_col3,#T_aab55_row4_col4,#T_aab55_row4_col5,#T_aab55_row4_col6,#T_aab55_row4_col7,#T_aab55_row4_col8,#T_aab55_row4_col9,#T_aab55_row4_col10,#T_aab55_row4_col11,#T_aab55_row4_col12,#T_aab55_row4_col13,#T_aab55_row4_col14,#T_aab55_row4_col15,#T_aab55_row4_col16,#T_aab55_row4_col17,#T_aab55_row4_col18,#T_aab55_row4_col19,#T_aab55_row4_col20,#T_aab55_row4_col21,#T_aab55_row4_col22,#T_aab55_row4_col23,#T_aab55_row4_col24,#T_aab55_row4_col25,#T_aab55_row4_col26,#T_aab55_row4_col27,#T_aab55_row5_col0,#T_aab55_row5_col1,#T_aab55_row5_col2,#T_aab55_row5_col3,#T_aab55_row5_col4,#T_aab55_row5_col5,#T_aab55_row5_col6,#T_aab55_row5_col7,#T_aab55_row5_col8,#T_aab55_row5_col9,#T_aab55_row5_col10,#T_aab55_row5_col11,#T_aab55_row5_col24,#T_aab55_row5_col25,#T_aab55_row5_col26,#T_aab55_row5_col27,#T_aab55_row6_col0,#T_aab55_row6_col1,#T_aab55_row6_col2,#T_aab55_row6_col3,#T_aab55_row6_col4,#T_aab55_row6_col5,#T_aab55_row6_col6,#T_aab55_row6_col7,#T_aab55_row6_col24,#T_aab55_row6_col25,#T_aab55_row6_col26,#T_aab55_row6_col27,#T_aab55_row7_col0,#T_aab55_row7_col1,#T_aab55_row7_col2,#T_aab55_row7_col3,#T_aab55_row7_col4,#T_aab55_row7_col5,#T_aab55_row7_col6,#T_aab55_row7_col23,#T_aab55_row7_col24,#T_aab55_row7_col25,#T_aab55_row7_col26,#T_aab55_row7_col27,#T_aab55_row8_col0,#T_aab55_row8_col1,#T_aab55_row8_col2,#T_aab55_row8_col3,#T_aab55_row8_col4,#T_aab55_row8_col5,#T_aab55_row8_col6,#T_aab55_row8_col18,#T_aab55_row8_col19,#T_aab55_row8_col20,#T_aab55_row8_col21,#T_aab55_row8_col22,#T_aab55_row8_col23,#T_aab55_row8_col24,#T_aab55_row8_col25,#T_aab55_row8_col26,#T_aab55_row8_col27,#T_aab55_row9_col0,#T_aab55_row9_col1,#T_aab55_row9_col2,#T_aab55_row9_col3,#T_aab55_row9_col4,#T_aab55_row9_col5,#T_aab55_row9_col6,#T_aab55_row9_col7,#T_aab55_row9_col15,#T_aab55_row9_col18,#T_aab55_row9_col19,#T_aab55_row9_col20,#T_aab55_row9_col21,#T_aab55_row9_col22,#T_aab55_row9_col23,#T_aab55_row9_col24,#T_aab55_row9_col25,#T_aab55_row9_col26,#T_aab55_row9_col27,#T_aab55_row10_col0,#T_aab55_row10_col1,#T_aab55_row10_col2,#T_aab55_row10_col3,#T_aab55_row10_col4,#T_aab55_row10_col5,#T_aab55_row10_col6,#T_aab55_row10_col7,#T_aab55_row10_col8,#T_aab55_row10_col14,#T_aab55_row10_col15,#T_aab55_row10_col16,#T_aab55_row10_col17,#T_aab55_row10_col18,#T_aab55_row10_col19,#T_aab55_row10_col20,#T_aab55_row10_col21,#T_aab55_row10_col22,#T_aab55_row10_col23,#T_aab55_row10_col24,#T_aab55_row10_col25,#T_aab55_row10_col26,#T_aab55_row10_col27,#T_aab55_row11_col0,#T_aab55_row11_col1,#T_aab55_row11_col2,#T_aab55_row11_col3,#T_aab55_row11_col4,#T_aab55_row11_col5,#T_aab55_row11_col6,#T_aab55_row11_col7,#T_aab55_row11_col8,#T_aab55_row11_col9,#T_aab55_row11_col10,#T_aab55_row11_col15,#T_aab55_row11_col16,#T_aab55_row11_col17,#T_aab55_row11_col18,#T_aab55_row11_col19,#T_aab55_row11_col20,#T_aab55_row11_col21,#T_aab55_row11_col22,#T_aab55_row11_col23,#T_aab55_row11_col24,#T_aab55_row11_col25,#T_aab55_row11_col26,#T_aab55_row11_col27,#T_aab55_row12_col0,#T_aab55_row12_col1,#T_aab55_row12_col2,#T_aab55_row12_col3,#T_aab55_row12_col4,#T_aab55_row12_col5,#T_aab55_row12_col6,#T_aab55_row12_col7,#T_aab55_row12_col8,#T_aab55_row12_col9,#T_aab55_row12_col10,#T_aab55_row12_col15,#T_aab55_row12_col16,#T_aab55_row12_col17,#T_aab55_row12_col18,#T_aab55_row12_col19,#T_aab55_row12_col20,#T_aab55_row12_col21,#T_aab55_row12_col22,#T_aab55_row12_col23,#T_aab55_row12_col24,#T_aab55_row12_col25,#T_aab55_row12_col26,#T_aab55_row12_col27,#T_aab55_row13_col0,#T_aab55_row13_col1,#T_aab55_row13_col2,#T_aab55_row13_col3,#T_aab55_row13_col4,#T_aab55_row13_col5,#T_aab55_row13_col6,#T_aab55_row13_col7,#T_aab55_row13_col8,#T_aab55_row13_col9,#T_aab55_row13_col10,#T_aab55_row13_col11,#T_aab55_row13_col18,#T_aab55_row13_col19,#T_aab55_row13_col20,#T_aab55_row13_col21,#T_aab55_row13_col22,#T_aab55_row13_col23,#T_aab55_row13_col24,#T_aab55_row13_col25,#T_aab55_row13_col26,#T_aab55_row13_col27,#T_aab55_row14_col0,#T_aab55_row14_col1,#T_aab55_row14_col2,#T_aab55_row14_col3,#T_aab55_row14_col4,#T_aab55_row14_col5,#T_aab55_row14_col6,#T_aab55_row14_col7,#T_aab55_row14_col8,#T_aab55_row14_col9,#T_aab55_row14_col10,#T_aab55_row14_col11,#T_aab55_row14_col12,#T_aab55_row14_col19,#T_aab55_row14_col20,#T_aab55_row14_col21,#T_aab55_row14_col22,#T_aab55_row14_col23,#T_aab55_row14_col24,#T_aab55_row14_col25,#T_aab55_row14_col26,#T_aab55_row14_col27,#T_aab55_row15_col0,#T_aab55_row15_col1,#T_aab55_row15_col2,#T_aab55_row15_col3,#T_aab55_row15_col4,#T_aab55_row15_col5,#T_aab55_row15_col6,#T_aab55_row15_col7,#T_aab55_row15_col8,#T_aab55_row15_col9,#T_aab55_row15_col10,#T_aab55_row15_col11,#T_aab55_row15_col12,#T_aab55_row15_col13,#T_aab55_row15_col20,#T_aab55_row15_col21,#T_aab55_row15_col22,#T_aab55_row15_col23,#T_aab55_row15_col24,#T_aab55_row15_col25,#T_aab55_row15_col26,#T_aab55_row15_col27,#T_aab55_row16_col0,#T_aab55_row16_col1,#T_aab55_row16_col2,#T_aab55_row16_col3,#T_aab55_row16_col4,#T_aab55_row16_col5,#T_aab55_row16_col6,#T_aab55_row16_col7,#T_aab55_row16_col8,#T_aab55_row16_col9,#T_aab55_row16_col10,#T_aab55_row16_col11,#T_aab55_row16_col12,#T_aab55_row16_col13,#T_aab55_row16_col14,#T_aab55_row16_col20,#T_aab55_row16_col21,#T_aab55_row16_col22,#T_aab55_row16_col23,#T_aab55_row16_col24,#T_aab55_row16_col25,#T_aab55_row16_col26,#T_aab55_row16_col27,#T_aab55_row17_col0,#T_aab55_row17_col1,#T_aab55_row17_col2,#T_aab55_row17_col3,#T_aab55_row17_col4,#T_aab55_row17_col5,#T_aab55_row17_col6,#T_aab55_row17_col7,#T_aab55_row17_col8,#T_aab55_row17_col9,#T_aab55_row17_col10,#T_aab55_row17_col11,#T_aab55_row17_col12,#T_aab55_row17_col13,#T_aab55_row17_col14,#T_aab55_row17_col15,#T_aab55_row17_col16,#T_aab55_row17_col21,#T_aab55_row17_col22,#T_aab55_row17_col23,#T_aab55_row17_col24,#T_aab55_row17_col25,#T_aab55_row17_col26,#T_aab55_row17_col27,#T_aab55_row18_col0,#T_aab55_row18_col1,#T_aab55_row18_col2,#T_aab55_row18_col3,#T_aab55_row18_col4,#T_aab55_row18_col5,#T_aab55_row18_col6,#T_aab55_row18_col7,#T_aab55_row18_col8,#T_aab55_row18_col9,#T_aab55_row18_col10,#T_aab55_row18_col11,#T_aab55_row18_col12,#T_aab55_row18_col13,#T_aab55_row18_col21,#T_aab55_row18_col22,#T_aab55_row18_col23,#T_aab55_row18_col24,#T_aab55_row18_col25,#T_aab55_row18_col26,#T_aab55_row18_col27,#T_aab55_row19_col0,#T_aab55_row19_col1,#T_aab55_row19_col2,#T_aab55_row19_col3,#T_aab55_row19_col4,#T_aab55_row19_col5,#T_aab55_row19_col6,#T_aab55_row19_col7,#T_aab55_row19_col8,#T_aab55_row19_col9,#T_aab55_row19_col10,#T_aab55_row19_col11,#T_aab55_row19_col20,#T_aab55_row19_col21,#T_aab55_row19_col22,#T_aab55_row19_col23,#T_aab55_row19_col24,#T_aab55_row19_col25,#T_aab55_row19_col26,#T_aab55_row19_col27,#T_aab55_row20_col0,#T_aab55_row20_col1,#T_aab55_row20_col2,#T_aab55_row20_col3,#T_aab55_row20_col4,#T_aab55_row20_col5,#T_aab55_row20_col6,#T_aab55_row20_col7,#T_aab55_row20_col8,#T_aab55_row20_col9,#T_aab55_row20_col19,#T_aab55_row20_col20,#T_aab55_row20_col21,#T_aab55_row20_col22,#T_aab55_row20_col23,#T_aab55_row20_col24,#T_aab55_row20_col25,#T_aab55_row20_col26,#T_aab55_row20_col27,#T_aab55_row21_col0,#T_aab55_row21_col1,#T_aab55_row21_col2,#T_aab55_row21_col3,#T_aab55_row21_col4,#T_aab55_row21_col5,#T_aab55_row21_col6,#T_aab55_row21_col7,#T_aab55_row21_col18,#T_aab55_row21_col19,#T_aab55_row21_col20,#T_aab55_row21_col21,#T_aab55_row21_col22,#T_aab55_row21_col23,#T_aab55_row21_col24,#T_aab55_row21_col25,#T_aab55_row21_col26,#T_aab55_row21_col27,#T_aab55_row22_col0,#T_aab55_row22_col1,#T_aab55_row22_col2,#T_aab55_row22_col3,#T_aab55_row22_col4,#T_aab55_row22_col5,#T_aab55_row22_col16,#T_aab55_row22_col17,#T_aab55_row22_col18,#T_aab55_row22_col19,#T_aab55_row22_col20,#T_aab55_row22_col21,#T_aab55_row22_col22,#T_aab55_row22_col23,#T_aab55_row22_col24,#T_aab55_row22_col25,#T_aab55_row22_col26,#T_aab55_row22_col27,#T_aab55_row23_col0,#T_aab55_row23_col1,#T_aab55_row23_col2,#T_aab55_row23_col3,#T_aab55_row23_col14,#T_aab55_row23_col15,#T_aab55_row23_col16,#T_aab55_row23_col17,#T_aab55_row23_col18,#T_aab55_row23_col19,#T_aab55_row23_col20,#T_aab55_row23_col21,#T_aab55_row23_col22,#T_aab55_row23_col23,#T_aab55_row23_col24,#T_aab55_row23_col25,#T_aab55_row23_col26,#T_aab55_row23_col27,#T_aab55_row24_col0,#T_aab55_row24_col1,#T_aab55_row24_col2,#T_aab55_row24_col3,#T_aab55_row24_col12,#T_aab55_row24_col13,#T_aab55_row24_col14,#T_aab55_row24_col15,#T_aab55_row24_col16,#T_aab55_row24_col17,#T_aab55_row24_col18,#T_aab55_row24_col19,#T_aab55_row24_col20,#T_aab55_row24_col21,#T_aab55_row24_col22,#T_aab55_row24_col23,#T_aab55_row24_col24,#T_aab55_row24_col25,#T_aab55_row24_col26,#T_aab55_row24_col27,#T_aab55_row25_col0,#T_aab55_row25_col1,#T_aab55_row25_col2,#T_aab55_row25_col3,#T_aab55_row25_col4,#T_aab55_row25_col5,#T_aab55_row25_col6,#T_aab55_row25_col7,#T_aab55_row25_col8,#T_aab55_row25_col9,#T_aab55_row25_col10,#T_aab55_row25_col11,#T_aab55_row25_col12,#T_aab55_row25_col13,#T_aab55_row25_col14,#T_aab55_row25_col15,#T_aab55_row25_col16,#T_aab55_row25_col17,#T_aab55_row25_col18,#T_aab55_row25_col19,#T_aab55_row25_col20,#T_aab55_row25_col21,#T_aab55_row25_col22,#T_aab55_row25_col23,#T_aab55_row25_col24,#T_aab55_row25_col25,#T_aab55_row25_col26,#T_aab55_row25_col27,#T_aab55_row26_col0,#T_aab55_row26_col1,#T_aab55_row26_col2,#T_aab55_row26_col3,#T_aab55_row26_col4,#T_aab55_row26_col5,#T_aab55_row26_col6,#T_aab55_row26_col7,#T_aab55_row26_col8,#T_aab55_row26_col9,#T_aab55_row26_col10,#T_aab55_row26_col11,#T_aab55_row26_col12,#T_aab55_row26_col13,#T_aab55_row26_col14,#T_aab55_row26_col15,#T_aab55_row26_col16,#T_aab55_row26_col17,#T_aab55_row26_col18,#T_aab55_row26_col19,#T_aab55_row26_col20,#T_aab55_row26_col21,#T_aab55_row26_col22,#T_aab55_row26_col23,#T_aab55_row26_col24,#T_aab55_row26_col25,#T_aab55_row26_col26,#T_aab55_row26_col27,#T_aab55_row27_col0,#T_aab55_row27_col1,#T_aab55_row27_col2,#T_aab55_row27_col3,#T_aab55_row27_col4,#T_aab55_row27_col5,#T_aab55_row27_col6,#T_aab55_row27_col7,#T_aab55_row27_col8,#T_aab55_row27_col9,#T_aab55_row27_col10,#T_aab55_row27_col11,#T_aab55_row27_col12,#T_aab55_row27_col13,#T_aab55_row27_col14,#T_aab55_row27_col15,#T_aab55_row27_col16,#T_aab55_row27_col17,#T_aab55_row27_col18,#T_aab55_row27_col19,#T_aab55_row27_col20,#T_aab55_row27_col21,#T_aab55_row27_col22,#T_aab55_row27_col23,#T_aab55_row27_col24,#T_aab55_row27_col25,#T_aab55_row27_col26,#T_aab55_row27_col27{
            font-size:  4pt;
            background-color:  #f7fcf5;
            color:  #000000;
        }#T_aab55_row5_col12{
            font-size:  4pt;
            background-color:  #f5fbf3;
            color:  #000000;
        }#T_aab55_row5_col13,#T_aab55_row5_col14,#T_aab55_row5_col15,#T_aab55_row8_col7,#T_aab55_row22_col6{
            font-size:  4pt;
            background-color:  #edf8e9;
            color:  #000000;
        }#T_aab55_row5_col16{
            font-size:  4pt;
            background-color:  #75c477;
            color:  #000000;
        }#T_aab55_row5_col17{
            font-size:  4pt;
            background-color:  #65bd6f;
            color:  #000000;
        }#T_aab55_row5_col18{
            font-size:  4pt;
            background-color:  #309950;
            color:  #000000;
        }#T_aab55_row5_col19{
            font-size:  4pt;
            background-color:  #e8f6e4;
            color:  #000000;
        }#T_aab55_row5_col20{
            font-size:  4pt;
            background-color:  #3aa357;
            color:  #000000;
        }#T_aab55_row5_col21,#T_aab55_row5_col22,#T_aab55_row5_col23,#T_aab55_row6_col13,#T_aab55_row6_col14,#T_aab55_row6_col15,#T_aab55_row6_col16,#T_aab55_row6_col17,#T_aab55_row6_col20,#T_aab55_row7_col9,#T_aab55_row7_col10,#T_aab55_row7_col11,#T_aab55_row7_col12,#T_aab55_row7_col13,#T_aab55_row7_col14,#T_aab55_row7_col15,#T_aab55_row7_col16,#T_aab55_row8_col9,#T_aab55_row8_col10,#T_aab55_row8_col11,#T_aab55_row8_col12,#T_aab55_row8_col13,#T_aab55_row9_col11,#T_aab55_row9_col12,#T_aab55_row10_col12,#T_aab55_row11_col12,#T_aab55_row12_col13,#T_aab55_row14_col15,#T_aab55_row14_col16,#T_aab55_row15_col16,#T_aab55_row15_col17,#T_aab55_row16_col18,#T_aab55_row17_col18,#T_aab55_row17_col19,#T_aab55_row18_col17,#T_aab55_row18_col18,#T_aab55_row19_col15,#T_aab55_row19_col16,#T_aab55_row19_col17,#T_aab55_row20_col13,#T_aab55_row20_col14,#T_aab55_row20_col15,#T_aab55_row20_col16,#T_aab55_row21_col11,#T_aab55_row21_col12,#T_aab55_row21_col13,#T_aab55_row21_col14,#T_aab55_row22_col9,#T_aab55_row22_col10,#T_aab55_row22_col11,#T_aab55_row22_col12,#T_aab55_row23_col7,#T_aab55_row23_col8,#T_aab55_row23_col9,#T_aab55_row23_col10,#T_aab55_row24_col4,#T_aab55_row24_col5,#T_aab55_row24_col6,#T_aab55_row24_col7{
            font-size:  4pt;
            background-color:  #00441b;
            color:  #f1f1f1;
        }#T_aab55_row6_col8{
            font-size:  4pt;
            background-color:  #e6f5e1;
            color:  #000000;
        }#T_aab55_row6_col9{
            font-size:  4pt;
            background-color:  #e1f3dc;
            color:  #000000;
        }#T_aab55_row6_col10{
            font-size:  4pt;
            background-color:  #a2d99c;
            color:  #000000;
        }#T_aab55_row6_col11,#T_aab55_row9_col17,#T_aab55_row10_col11{
            font-size:  4pt;
            background-color:  #48ae60;
            color:  #000000;
        }#T_aab55_row6_col12{
            font-size:  4pt;
            background-color:  #359e53;
            color:  #000000;
        }#T_aab55_row6_col18,#T_aab55_row13_col14{
            font-size:  4pt;
            background-color:  #00682a;
            color:  #f1f1f1;
        }#T_aab55_row6_col19{
            font-size:  4pt;
            background-color:  #319a50;
            color:  #000000;
        }#T_aab55_row6_col21,#T_aab55_row14_col14{
            font-size:  4pt;
            background-color:  #005522;
            color:  #f1f1f1;
        }#T_aab55_row6_col22{
            font-size:  4pt;
            background-color:  #17813d;
            color:  #f1f1f1;
        }#T_aab55_row6_col23{
            font-size:  4pt;
            background-color:  #72c375;
            color:  #000000;
        }#T_aab55_row7_col7{
            font-size:  4pt;
            background-color:  #d5efcf;
            color:  #000000;
        }#T_aab55_row7_col8{
            font-size:  4pt;
            background-color:  #005723;
            color:  #f1f1f1;
        }#T_aab55_row7_col17{
            font-size:  4pt;
            background-color:  #00471c;
            color:  #f1f1f1;
        }#T_aab55_row7_col18,#T_aab55_row16_col16{
            font-size:  4pt;
            background-color:  #a3da9d;
            color:  #000000;
        }#T_aab55_row7_col19{
            font-size:  4pt;
            background-color:  #afdfa8;
            color:  #000000;
        }#T_aab55_row7_col20{
            font-size:  4pt;
            background-color:  #b1e0ab;
            color:  #000000;
        }#T_aab55_row7_col21{
            font-size:  4pt;
            background-color:  #ceecc8;
            color:  #000000;
        }#T_aab55_row7_col22{
            font-size:  4pt;
            background-color:  #ddf2d8;
            color:  #000000;
        }#T_aab55_row8_col8,#T_aab55_row22_col8{
            font-size:  4pt;
            background-color:  #026f2e;
            color:  #f1f1f1;
        }#T_aab55_row8_col14,#T_aab55_row21_col15{
            font-size:  4pt;
            background-color:  #19833e;
            color:  #f1f1f1;
        }#T_aab55_row8_col15{
            font-size:  4pt;
            background-color:  #2a924a;
            color:  #000000;
        }#T_aab55_row8_col16{
            font-size:  4pt;
            background-color:  #004c1e;
            color:  #f1f1f1;
        }#T_aab55_row8_col17,#T_aab55_row13_col13{
            font-size:  4pt;
            background-color:  #005321;
            color:  #f1f1f1;
        }#T_aab55_row9_col8,#T_aab55_row22_col14{
            font-size:  4pt;
            background-color:  #b4e1ad;
            color:  #000000;
        }#T_aab55_row9_col9{
            font-size:  4pt;
            background-color:  #45ad5f;
            color:  #000000;
        }#T_aab55_row9_col10{
            font-size:  4pt;
            background-color:  #90d18d;
            color:  #000000;
        }#T_aab55_row9_col13{
            font-size:  4pt;
            background-color:  #127c39;
            color:  #f1f1f1;
        }#T_aab55_row9_col14,#T_aab55_row12_col11,#T_aab55_row23_col13{
            font-size:  4pt;
            background-color:  #f1faee;
            color:  #000000;
        }#T_aab55_row9_col16{
            font-size:  4pt;
            background-color:  #dbf1d5;
            color:  #000000;
        }#T_aab55_row10_col9{
            font-size:  4pt;
            background-color:  #eff9ec;
            color:  #000000;
        }#T_aab55_row10_col10,#T_aab55_row11_col14,#T_aab55_row13_col17,#T_aab55_row18_col20,#T_aab55_row21_col17{
            font-size:  4pt;
            background-color:  #f6fcf4;
            color:  #000000;
        }#T_aab55_row10_col13{
            font-size:  4pt;
            background-color:  #a7dba0;
            color:  #000000;
        }#T_aab55_row11_col11{
            font-size:  4pt;
            background-color:  #60ba6c;
            color:  #000000;
        }#T_aab55_row11_col13,#T_aab55_row12_col12,#T_aab55_row16_col19{
            font-size:  4pt;
            background-color:  #228a44;
            color:  #000000;
        }#T_aab55_row12_col14{
            font-size:  4pt;
            background-color:  #c0e6b9;
            color:  #000000;
        }#T_aab55_row13_col12{
            font-size:  4pt;
            background-color:  #e2f4dd;
            color:  #000000;
        }#T_aab55_row13_col15{
            font-size:  4pt;
            background-color:  #3fa95c;
            color:  #000000;
        }#T_aab55_row13_col16{
            font-size:  4pt;
            background-color:  #8ed08b;
            color:  #000000;
        }#T_aab55_row14_col13,#T_aab55_row21_col16{
            font-size:  4pt;
            background-color:  #b2e0ac;
            color:  #000000;
        }#T_aab55_row14_col17{
            font-size:  4pt;
            background-color:  #7fc97f;
            color:  #000000;
        }#T_aab55_row14_col18,#T_aab55_row20_col10{
            font-size:  4pt;
            background-color:  #e9f7e5;
            color:  #000000;
        }#T_aab55_row15_col14{
            font-size:  4pt;
            background-color:  #d9f0d3;
            color:  #000000;
        }#T_aab55_row15_col15{
            font-size:  4pt;
            background-color:  #268e47;
            color:  #000000;
        }#T_aab55_row15_col18{
            font-size:  4pt;
            background-color:  #4eb264;
            color:  #000000;
        }#T_aab55_row15_col19{
            font-size:  4pt;
            background-color:  #e8f6e3;
            color:  #000000;
        }#T_aab55_row16_col15,#T_aab55_row24_col11{
            font-size:  4pt;
            background-color:  #eef8ea;
            color:  #000000;
        }#T_aab55_row16_col17{
            font-size:  4pt;
            background-color:  #00451c;
            color:  #f1f1f1;
        }#T_aab55_row17_col17{
            font-size:  4pt;
            background-color:  #00491d;
            color:  #f1f1f1;
        }#T_aab55_row17_col20{
            font-size:  4pt;
            background-color:  #c7e9c0;
            color:  #000000;
        }#T_aab55_row18_col14{
            font-size:  4pt;
            background-color:  #d8f0d2;
            color:  #000000;
        }#T_aab55_row18_col15{
            font-size:  4pt;
            background-color:  #6ec173;
            color:  #000000;
        }#T_aab55_row18_col16{
            font-size:  4pt;
            background-color:  #29914a;
            color:  #000000;
        }#T_aab55_row18_col19{
            font-size:  4pt;
            background-color:  #0c7735;
            color:  #f1f1f1;
        }#T_aab55_row19_col12{
            font-size:  4pt;
            background-color:  #def2d9;
            color:  #000000;
        }#T_aab55_row19_col13{
            font-size:  4pt;
            background-color:  #52b365;
            color:  #000000;
        }#T_aab55_row19_col14{
            font-size:  4pt;
            background-color:  #006328;
            color:  #f1f1f1;
        }#T_aab55_row19_col18{
            font-size:  4pt;
            background-color:  #00481d;
            color:  #f1f1f1;
        }#T_aab55_row19_col19{
            font-size:  4pt;
            background-color:  #278f48;
            color:  #000000;
        }#T_aab55_row20_col11{
            font-size:  4pt;
            background-color:  #86cc85;
            color:  #000000;
        }#T_aab55_row20_col12{
            font-size:  4pt;
            background-color:  #006d2c;
            color:  #f1f1f1;
        }#T_aab55_row20_col17{
            font-size:  4pt;
            background-color:  #16803c;
            color:  #f1f1f1;
        }#T_aab55_row20_col18{
            font-size:  4pt;
            background-color:  #b6e2af;
            color:  #000000;
        }#T_aab55_row21_col8{
            font-size:  4pt;
            background-color:  #eaf7e6;
            color:  #000000;
        }#T_aab55_row21_col9{
            font-size:  4pt;
            background-color:  #c4e8bd;
            color:  #000000;
        }#T_aab55_row21_col10{
            font-size:  4pt;
            background-color:  #097532;
            color:  #f1f1f1;
        }#T_aab55_row22_col7{
            font-size:  4pt;
            background-color:  #349d53;
            color:  #000000;
        }#T_aab55_row22_col13{
            font-size:  4pt;
            background-color:  #1d8640;
            color:  #000000;
        }#T_aab55_row22_col15{
            font-size:  4pt;
            background-color:  #f2faef;
            color:  #000000;
        }#T_aab55_row23_col4{
            font-size:  4pt;
            background-color:  #97d492;
            color:  #000000;
        }#T_aab55_row23_col5{
            font-size:  4pt;
            background-color:  #339c52;
            color:  #000000;
        }#T_aab55_row23_col6{
            font-size:  4pt;
            background-color:  #006729;
            color:  #f1f1f1;
        }#T_aab55_row23_col11{
            font-size:  4pt;
            background-color:  #005020;
            color:  #f1f1f1;
        }#T_aab55_row23_col12{
            font-size:  4pt;
            background-color:  #6abf71;
            color:  #000000;
        }#T_aab55_row24_col8{
            font-size:  4pt;
            background-color:  #0a7633;
            color:  #f1f1f1;
        }#T_aab55_row24_col9{
            font-size:  4pt;
            background-color:  #66bd6f;
            color:  #000000;
        }#T_aab55_row24_col10{
            font-size:  4pt;
            background-color:  #6bc072;
            color:  #000000;
        }</style><table id="T_aab55_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>        <th class="col_heading level0 col10" >10</th>        <th class="col_heading level0 col11" >11</th>        <th class="col_heading level0 col12" >12</th>        <th class="col_heading level0 col13" >13</th>        <th class="col_heading level0 col14" >14</th>        <th class="col_heading level0 col15" >15</th>        <th class="col_heading level0 col16" >16</th>        <th class="col_heading level0 col17" >17</th>        <th class="col_heading level0 col18" >18</th>        <th class="col_heading level0 col19" >19</th>        <th class="col_heading level0 col20" >20</th>        <th class="col_heading level0 col21" >21</th>        <th class="col_heading level0 col22" >22</th>        <th class="col_heading level0 col23" >23</th>        <th class="col_heading level0 col24" >24</th>        <th class="col_heading level0 col25" >25</th>        <th class="col_heading level0 col26" >26</th>        <th class="col_heading level0 col27" >27</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_aab55_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_aab55_row0_col0" class="data row0 col0" >0</td>
                        <td id="T_aab55_row0_col1" class="data row0 col1" >0</td>
                        <td id="T_aab55_row0_col2" class="data row0 col2" >0</td>
                        <td id="T_aab55_row0_col3" class="data row0 col3" >0</td>
                        <td id="T_aab55_row0_col4" class="data row0 col4" >0</td>
                        <td id="T_aab55_row0_col5" class="data row0 col5" >0</td>
                        <td id="T_aab55_row0_col6" class="data row0 col6" >0</td>
                        <td id="T_aab55_row0_col7" class="data row0 col7" >0</td>
                        <td id="T_aab55_row0_col8" class="data row0 col8" >0</td>
                        <td id="T_aab55_row0_col9" class="data row0 col9" >0</td>
                        <td id="T_aab55_row0_col10" class="data row0 col10" >0</td>
                        <td id="T_aab55_row0_col11" class="data row0 col11" >0</td>
                        <td id="T_aab55_row0_col12" class="data row0 col12" >0</td>
                        <td id="T_aab55_row0_col13" class="data row0 col13" >0</td>
                        <td id="T_aab55_row0_col14" class="data row0 col14" >0</td>
                        <td id="T_aab55_row0_col15" class="data row0 col15" >0</td>
                        <td id="T_aab55_row0_col16" class="data row0 col16" >0</td>
                        <td id="T_aab55_row0_col17" class="data row0 col17" >0</td>
                        <td id="T_aab55_row0_col18" class="data row0 col18" >0</td>
                        <td id="T_aab55_row0_col19" class="data row0 col19" >0</td>
                        <td id="T_aab55_row0_col20" class="data row0 col20" >0</td>
                        <td id="T_aab55_row0_col21" class="data row0 col21" >0</td>
                        <td id="T_aab55_row0_col22" class="data row0 col22" >0</td>
                        <td id="T_aab55_row0_col23" class="data row0 col23" >0</td>
                        <td id="T_aab55_row0_col24" class="data row0 col24" >0</td>
                        <td id="T_aab55_row0_col25" class="data row0 col25" >0</td>
                        <td id="T_aab55_row0_col26" class="data row0 col26" >0</td>
                        <td id="T_aab55_row0_col27" class="data row0 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_aab55_row1_col0" class="data row1 col0" >0</td>
                        <td id="T_aab55_row1_col1" class="data row1 col1" >0</td>
                        <td id="T_aab55_row1_col2" class="data row1 col2" >0</td>
                        <td id="T_aab55_row1_col3" class="data row1 col3" >0</td>
                        <td id="T_aab55_row1_col4" class="data row1 col4" >0</td>
                        <td id="T_aab55_row1_col5" class="data row1 col5" >0</td>
                        <td id="T_aab55_row1_col6" class="data row1 col6" >0</td>
                        <td id="T_aab55_row1_col7" class="data row1 col7" >0</td>
                        <td id="T_aab55_row1_col8" class="data row1 col8" >0</td>
                        <td id="T_aab55_row1_col9" class="data row1 col9" >0</td>
                        <td id="T_aab55_row1_col10" class="data row1 col10" >0</td>
                        <td id="T_aab55_row1_col11" class="data row1 col11" >0</td>
                        <td id="T_aab55_row1_col12" class="data row1 col12" >0</td>
                        <td id="T_aab55_row1_col13" class="data row1 col13" >0</td>
                        <td id="T_aab55_row1_col14" class="data row1 col14" >0</td>
                        <td id="T_aab55_row1_col15" class="data row1 col15" >0</td>
                        <td id="T_aab55_row1_col16" class="data row1 col16" >0</td>
                        <td id="T_aab55_row1_col17" class="data row1 col17" >0</td>
                        <td id="T_aab55_row1_col18" class="data row1 col18" >0</td>
                        <td id="T_aab55_row1_col19" class="data row1 col19" >0</td>
                        <td id="T_aab55_row1_col20" class="data row1 col20" >0</td>
                        <td id="T_aab55_row1_col21" class="data row1 col21" >0</td>
                        <td id="T_aab55_row1_col22" class="data row1 col22" >0</td>
                        <td id="T_aab55_row1_col23" class="data row1 col23" >0</td>
                        <td id="T_aab55_row1_col24" class="data row1 col24" >0</td>
                        <td id="T_aab55_row1_col25" class="data row1 col25" >0</td>
                        <td id="T_aab55_row1_col26" class="data row1 col26" >0</td>
                        <td id="T_aab55_row1_col27" class="data row1 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_aab55_row2_col0" class="data row2 col0" >0</td>
                        <td id="T_aab55_row2_col1" class="data row2 col1" >0</td>
                        <td id="T_aab55_row2_col2" class="data row2 col2" >0</td>
                        <td id="T_aab55_row2_col3" class="data row2 col3" >0</td>
                        <td id="T_aab55_row2_col4" class="data row2 col4" >0</td>
                        <td id="T_aab55_row2_col5" class="data row2 col5" >0</td>
                        <td id="T_aab55_row2_col6" class="data row2 col6" >0</td>
                        <td id="T_aab55_row2_col7" class="data row2 col7" >0</td>
                        <td id="T_aab55_row2_col8" class="data row2 col8" >0</td>
                        <td id="T_aab55_row2_col9" class="data row2 col9" >0</td>
                        <td id="T_aab55_row2_col10" class="data row2 col10" >0</td>
                        <td id="T_aab55_row2_col11" class="data row2 col11" >0</td>
                        <td id="T_aab55_row2_col12" class="data row2 col12" >0</td>
                        <td id="T_aab55_row2_col13" class="data row2 col13" >0</td>
                        <td id="T_aab55_row2_col14" class="data row2 col14" >0</td>
                        <td id="T_aab55_row2_col15" class="data row2 col15" >0</td>
                        <td id="T_aab55_row2_col16" class="data row2 col16" >0</td>
                        <td id="T_aab55_row2_col17" class="data row2 col17" >0</td>
                        <td id="T_aab55_row2_col18" class="data row2 col18" >0</td>
                        <td id="T_aab55_row2_col19" class="data row2 col19" >0</td>
                        <td id="T_aab55_row2_col20" class="data row2 col20" >0</td>
                        <td id="T_aab55_row2_col21" class="data row2 col21" >0</td>
                        <td id="T_aab55_row2_col22" class="data row2 col22" >0</td>
                        <td id="T_aab55_row2_col23" class="data row2 col23" >0</td>
                        <td id="T_aab55_row2_col24" class="data row2 col24" >0</td>
                        <td id="T_aab55_row2_col25" class="data row2 col25" >0</td>
                        <td id="T_aab55_row2_col26" class="data row2 col26" >0</td>
                        <td id="T_aab55_row2_col27" class="data row2 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_aab55_row3_col0" class="data row3 col0" >0</td>
                        <td id="T_aab55_row3_col1" class="data row3 col1" >0</td>
                        <td id="T_aab55_row3_col2" class="data row3 col2" >0</td>
                        <td id="T_aab55_row3_col3" class="data row3 col3" >0</td>
                        <td id="T_aab55_row3_col4" class="data row3 col4" >0</td>
                        <td id="T_aab55_row3_col5" class="data row3 col5" >0</td>
                        <td id="T_aab55_row3_col6" class="data row3 col6" >0</td>
                        <td id="T_aab55_row3_col7" class="data row3 col7" >0</td>
                        <td id="T_aab55_row3_col8" class="data row3 col8" >0</td>
                        <td id="T_aab55_row3_col9" class="data row3 col9" >0</td>
                        <td id="T_aab55_row3_col10" class="data row3 col10" >0</td>
                        <td id="T_aab55_row3_col11" class="data row3 col11" >0</td>
                        <td id="T_aab55_row3_col12" class="data row3 col12" >0</td>
                        <td id="T_aab55_row3_col13" class="data row3 col13" >0</td>
                        <td id="T_aab55_row3_col14" class="data row3 col14" >0</td>
                        <td id="T_aab55_row3_col15" class="data row3 col15" >0</td>
                        <td id="T_aab55_row3_col16" class="data row3 col16" >0</td>
                        <td id="T_aab55_row3_col17" class="data row3 col17" >0</td>
                        <td id="T_aab55_row3_col18" class="data row3 col18" >0</td>
                        <td id="T_aab55_row3_col19" class="data row3 col19" >0</td>
                        <td id="T_aab55_row3_col20" class="data row3 col20" >0</td>
                        <td id="T_aab55_row3_col21" class="data row3 col21" >0</td>
                        <td id="T_aab55_row3_col22" class="data row3 col22" >0</td>
                        <td id="T_aab55_row3_col23" class="data row3 col23" >0</td>
                        <td id="T_aab55_row3_col24" class="data row3 col24" >0</td>
                        <td id="T_aab55_row3_col25" class="data row3 col25" >0</td>
                        <td id="T_aab55_row3_col26" class="data row3 col26" >0</td>
                        <td id="T_aab55_row3_col27" class="data row3 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_aab55_row4_col0" class="data row4 col0" >0</td>
                        <td id="T_aab55_row4_col1" class="data row4 col1" >0</td>
                        <td id="T_aab55_row4_col2" class="data row4 col2" >0</td>
                        <td id="T_aab55_row4_col3" class="data row4 col3" >0</td>
                        <td id="T_aab55_row4_col4" class="data row4 col4" >0</td>
                        <td id="T_aab55_row4_col5" class="data row4 col5" >0</td>
                        <td id="T_aab55_row4_col6" class="data row4 col6" >0</td>
                        <td id="T_aab55_row4_col7" class="data row4 col7" >0</td>
                        <td id="T_aab55_row4_col8" class="data row4 col8" >0</td>
                        <td id="T_aab55_row4_col9" class="data row4 col9" >0</td>
                        <td id="T_aab55_row4_col10" class="data row4 col10" >0</td>
                        <td id="T_aab55_row4_col11" class="data row4 col11" >0</td>
                        <td id="T_aab55_row4_col12" class="data row4 col12" >0</td>
                        <td id="T_aab55_row4_col13" class="data row4 col13" >0</td>
                        <td id="T_aab55_row4_col14" class="data row4 col14" >0</td>
                        <td id="T_aab55_row4_col15" class="data row4 col15" >0</td>
                        <td id="T_aab55_row4_col16" class="data row4 col16" >0</td>
                        <td id="T_aab55_row4_col17" class="data row4 col17" >0</td>
                        <td id="T_aab55_row4_col18" class="data row4 col18" >0</td>
                        <td id="T_aab55_row4_col19" class="data row4 col19" >0</td>
                        <td id="T_aab55_row4_col20" class="data row4 col20" >0</td>
                        <td id="T_aab55_row4_col21" class="data row4 col21" >0</td>
                        <td id="T_aab55_row4_col22" class="data row4 col22" >0</td>
                        <td id="T_aab55_row4_col23" class="data row4 col23" >0</td>
                        <td id="T_aab55_row4_col24" class="data row4 col24" >0</td>
                        <td id="T_aab55_row4_col25" class="data row4 col25" >0</td>
                        <td id="T_aab55_row4_col26" class="data row4 col26" >0</td>
                        <td id="T_aab55_row4_col27" class="data row4 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_aab55_row5_col0" class="data row5 col0" >0</td>
                        <td id="T_aab55_row5_col1" class="data row5 col1" >0</td>
                        <td id="T_aab55_row5_col2" class="data row5 col2" >0</td>
                        <td id="T_aab55_row5_col3" class="data row5 col3" >0</td>
                        <td id="T_aab55_row5_col4" class="data row5 col4" >0</td>
                        <td id="T_aab55_row5_col5" class="data row5 col5" >0</td>
                        <td id="T_aab55_row5_col6" class="data row5 col6" >0</td>
                        <td id="T_aab55_row5_col7" class="data row5 col7" >0</td>
                        <td id="T_aab55_row5_col8" class="data row5 col8" >0</td>
                        <td id="T_aab55_row5_col9" class="data row5 col9" >0</td>
                        <td id="T_aab55_row5_col10" class="data row5 col10" >0</td>
                        <td id="T_aab55_row5_col11" class="data row5 col11" >0</td>
                        <td id="T_aab55_row5_col12" class="data row5 col12" >3</td>
                        <td id="T_aab55_row5_col13" class="data row5 col13" >18</td>
                        <td id="T_aab55_row5_col14" class="data row5 col14" >18</td>
                        <td id="T_aab55_row5_col15" class="data row5 col15" >18</td>
                        <td id="T_aab55_row5_col16" class="data row5 col16" >126</td>
                        <td id="T_aab55_row5_col17" class="data row5 col17" >136</td>
                        <td id="T_aab55_row5_col18" class="data row5 col18" >175</td>
                        <td id="T_aab55_row5_col19" class="data row5 col19" >26</td>
                        <td id="T_aab55_row5_col20" class="data row5 col20" >166</td>
                        <td id="T_aab55_row5_col21" class="data row5 col21" >255</td>
                        <td id="T_aab55_row5_col22" class="data row5 col22" >247</td>
                        <td id="T_aab55_row5_col23" class="data row5 col23" >127</td>
                        <td id="T_aab55_row5_col24" class="data row5 col24" >0</td>
                        <td id="T_aab55_row5_col25" class="data row5 col25" >0</td>
                        <td id="T_aab55_row5_col26" class="data row5 col26" >0</td>
                        <td id="T_aab55_row5_col27" class="data row5 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_aab55_row6_col0" class="data row6 col0" >0</td>
                        <td id="T_aab55_row6_col1" class="data row6 col1" >0</td>
                        <td id="T_aab55_row6_col2" class="data row6 col2" >0</td>
                        <td id="T_aab55_row6_col3" class="data row6 col3" >0</td>
                        <td id="T_aab55_row6_col4" class="data row6 col4" >0</td>
                        <td id="T_aab55_row6_col5" class="data row6 col5" >0</td>
                        <td id="T_aab55_row6_col6" class="data row6 col6" >0</td>
                        <td id="T_aab55_row6_col7" class="data row6 col7" >0</td>
                        <td id="T_aab55_row6_col8" class="data row6 col8" >30</td>
                        <td id="T_aab55_row6_col9" class="data row6 col9" >36</td>
                        <td id="T_aab55_row6_col10" class="data row6 col10" >94</td>
                        <td id="T_aab55_row6_col11" class="data row6 col11" >154</td>
                        <td id="T_aab55_row6_col12" class="data row6 col12" >170</td>
                        <td id="T_aab55_row6_col13" class="data row6 col13" >253</td>
                        <td id="T_aab55_row6_col14" class="data row6 col14" >253</td>
                        <td id="T_aab55_row6_col15" class="data row6 col15" >253</td>
                        <td id="T_aab55_row6_col16" class="data row6 col16" >253</td>
                        <td id="T_aab55_row6_col17" class="data row6 col17" >253</td>
                        <td id="T_aab55_row6_col18" class="data row6 col18" >225</td>
                        <td id="T_aab55_row6_col19" class="data row6 col19" >172</td>
                        <td id="T_aab55_row6_col20" class="data row6 col20" >253</td>
                        <td id="T_aab55_row6_col21" class="data row6 col21" >242</td>
                        <td id="T_aab55_row6_col22" class="data row6 col22" >195</td>
                        <td id="T_aab55_row6_col23" class="data row6 col23" >64</td>
                        <td id="T_aab55_row6_col24" class="data row6 col24" >0</td>
                        <td id="T_aab55_row6_col25" class="data row6 col25" >0</td>
                        <td id="T_aab55_row6_col26" class="data row6 col26" >0</td>
                        <td id="T_aab55_row6_col27" class="data row6 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_aab55_row7_col0" class="data row7 col0" >0</td>
                        <td id="T_aab55_row7_col1" class="data row7 col1" >0</td>
                        <td id="T_aab55_row7_col2" class="data row7 col2" >0</td>
                        <td id="T_aab55_row7_col3" class="data row7 col3" >0</td>
                        <td id="T_aab55_row7_col4" class="data row7 col4" >0</td>
                        <td id="T_aab55_row7_col5" class="data row7 col5" >0</td>
                        <td id="T_aab55_row7_col6" class="data row7 col6" >0</td>
                        <td id="T_aab55_row7_col7" class="data row7 col7" >49</td>
                        <td id="T_aab55_row7_col8" class="data row7 col8" >238</td>
                        <td id="T_aab55_row7_col9" class="data row7 col9" >253</td>
                        <td id="T_aab55_row7_col10" class="data row7 col10" >253</td>
                        <td id="T_aab55_row7_col11" class="data row7 col11" >253</td>
                        <td id="T_aab55_row7_col12" class="data row7 col12" >253</td>
                        <td id="T_aab55_row7_col13" class="data row7 col13" >253</td>
                        <td id="T_aab55_row7_col14" class="data row7 col14" >253</td>
                        <td id="T_aab55_row7_col15" class="data row7 col15" >253</td>
                        <td id="T_aab55_row7_col16" class="data row7 col16" >253</td>
                        <td id="T_aab55_row7_col17" class="data row7 col17" >251</td>
                        <td id="T_aab55_row7_col18" class="data row7 col18" >93</td>
                        <td id="T_aab55_row7_col19" class="data row7 col19" >82</td>
                        <td id="T_aab55_row7_col20" class="data row7 col20" >82</td>
                        <td id="T_aab55_row7_col21" class="data row7 col21" >56</td>
                        <td id="T_aab55_row7_col22" class="data row7 col22" >39</td>
                        <td id="T_aab55_row7_col23" class="data row7 col23" >0</td>
                        <td id="T_aab55_row7_col24" class="data row7 col24" >0</td>
                        <td id="T_aab55_row7_col25" class="data row7 col25" >0</td>
                        <td id="T_aab55_row7_col26" class="data row7 col26" >0</td>
                        <td id="T_aab55_row7_col27" class="data row7 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_aab55_row8_col0" class="data row8 col0" >0</td>
                        <td id="T_aab55_row8_col1" class="data row8 col1" >0</td>
                        <td id="T_aab55_row8_col2" class="data row8 col2" >0</td>
                        <td id="T_aab55_row8_col3" class="data row8 col3" >0</td>
                        <td id="T_aab55_row8_col4" class="data row8 col4" >0</td>
                        <td id="T_aab55_row8_col5" class="data row8 col5" >0</td>
                        <td id="T_aab55_row8_col6" class="data row8 col6" >0</td>
                        <td id="T_aab55_row8_col7" class="data row8 col7" >18</td>
                        <td id="T_aab55_row8_col8" class="data row8 col8" >219</td>
                        <td id="T_aab55_row8_col9" class="data row8 col9" >253</td>
                        <td id="T_aab55_row8_col10" class="data row8 col10" >253</td>
                        <td id="T_aab55_row8_col11" class="data row8 col11" >253</td>
                        <td id="T_aab55_row8_col12" class="data row8 col12" >253</td>
                        <td id="T_aab55_row8_col13" class="data row8 col13" >253</td>
                        <td id="T_aab55_row8_col14" class="data row8 col14" >198</td>
                        <td id="T_aab55_row8_col15" class="data row8 col15" >182</td>
                        <td id="T_aab55_row8_col16" class="data row8 col16" >247</td>
                        <td id="T_aab55_row8_col17" class="data row8 col17" >241</td>
                        <td id="T_aab55_row8_col18" class="data row8 col18" >0</td>
                        <td id="T_aab55_row8_col19" class="data row8 col19" >0</td>
                        <td id="T_aab55_row8_col20" class="data row8 col20" >0</td>
                        <td id="T_aab55_row8_col21" class="data row8 col21" >0</td>
                        <td id="T_aab55_row8_col22" class="data row8 col22" >0</td>
                        <td id="T_aab55_row8_col23" class="data row8 col23" >0</td>
                        <td id="T_aab55_row8_col24" class="data row8 col24" >0</td>
                        <td id="T_aab55_row8_col25" class="data row8 col25" >0</td>
                        <td id="T_aab55_row8_col26" class="data row8 col26" >0</td>
                        <td id="T_aab55_row8_col27" class="data row8 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_aab55_row9_col0" class="data row9 col0" >0</td>
                        <td id="T_aab55_row9_col1" class="data row9 col1" >0</td>
                        <td id="T_aab55_row9_col2" class="data row9 col2" >0</td>
                        <td id="T_aab55_row9_col3" class="data row9 col3" >0</td>
                        <td id="T_aab55_row9_col4" class="data row9 col4" >0</td>
                        <td id="T_aab55_row9_col5" class="data row9 col5" >0</td>
                        <td id="T_aab55_row9_col6" class="data row9 col6" >0</td>
                        <td id="T_aab55_row9_col7" class="data row9 col7" >0</td>
                        <td id="T_aab55_row9_col8" class="data row9 col8" >80</td>
                        <td id="T_aab55_row9_col9" class="data row9 col9" >156</td>
                        <td id="T_aab55_row9_col10" class="data row9 col10" >107</td>
                        <td id="T_aab55_row9_col11" class="data row9 col11" >253</td>
                        <td id="T_aab55_row9_col12" class="data row9 col12" >253</td>
                        <td id="T_aab55_row9_col13" class="data row9 col13" >205</td>
                        <td id="T_aab55_row9_col14" class="data row9 col14" >11</td>
                        <td id="T_aab55_row9_col15" class="data row9 col15" >0</td>
                        <td id="T_aab55_row9_col16" class="data row9 col16" >43</td>
                        <td id="T_aab55_row9_col17" class="data row9 col17" >154</td>
                        <td id="T_aab55_row9_col18" class="data row9 col18" >0</td>
                        <td id="T_aab55_row9_col19" class="data row9 col19" >0</td>
                        <td id="T_aab55_row9_col20" class="data row9 col20" >0</td>
                        <td id="T_aab55_row9_col21" class="data row9 col21" >0</td>
                        <td id="T_aab55_row9_col22" class="data row9 col22" >0</td>
                        <td id="T_aab55_row9_col23" class="data row9 col23" >0</td>
                        <td id="T_aab55_row9_col24" class="data row9 col24" >0</td>
                        <td id="T_aab55_row9_col25" class="data row9 col25" >0</td>
                        <td id="T_aab55_row9_col26" class="data row9 col26" >0</td>
                        <td id="T_aab55_row9_col27" class="data row9 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_aab55_row10_col0" class="data row10 col0" >0</td>
                        <td id="T_aab55_row10_col1" class="data row10 col1" >0</td>
                        <td id="T_aab55_row10_col2" class="data row10 col2" >0</td>
                        <td id="T_aab55_row10_col3" class="data row10 col3" >0</td>
                        <td id="T_aab55_row10_col4" class="data row10 col4" >0</td>
                        <td id="T_aab55_row10_col5" class="data row10 col5" >0</td>
                        <td id="T_aab55_row10_col6" class="data row10 col6" >0</td>
                        <td id="T_aab55_row10_col7" class="data row10 col7" >0</td>
                        <td id="T_aab55_row10_col8" class="data row10 col8" >0</td>
                        <td id="T_aab55_row10_col9" class="data row10 col9" >14</td>
                        <td id="T_aab55_row10_col10" class="data row10 col10" >1</td>
                        <td id="T_aab55_row10_col11" class="data row10 col11" >154</td>
                        <td id="T_aab55_row10_col12" class="data row10 col12" >253</td>
                        <td id="T_aab55_row10_col13" class="data row10 col13" >90</td>
                        <td id="T_aab55_row10_col14" class="data row10 col14" >0</td>
                        <td id="T_aab55_row10_col15" class="data row10 col15" >0</td>
                        <td id="T_aab55_row10_col16" class="data row10 col16" >0</td>
                        <td id="T_aab55_row10_col17" class="data row10 col17" >0</td>
                        <td id="T_aab55_row10_col18" class="data row10 col18" >0</td>
                        <td id="T_aab55_row10_col19" class="data row10 col19" >0</td>
                        <td id="T_aab55_row10_col20" class="data row10 col20" >0</td>
                        <td id="T_aab55_row10_col21" class="data row10 col21" >0</td>
                        <td id="T_aab55_row10_col22" class="data row10 col22" >0</td>
                        <td id="T_aab55_row10_col23" class="data row10 col23" >0</td>
                        <td id="T_aab55_row10_col24" class="data row10 col24" >0</td>
                        <td id="T_aab55_row10_col25" class="data row10 col25" >0</td>
                        <td id="T_aab55_row10_col26" class="data row10 col26" >0</td>
                        <td id="T_aab55_row10_col27" class="data row10 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_aab55_row11_col0" class="data row11 col0" >0</td>
                        <td id="T_aab55_row11_col1" class="data row11 col1" >0</td>
                        <td id="T_aab55_row11_col2" class="data row11 col2" >0</td>
                        <td id="T_aab55_row11_col3" class="data row11 col3" >0</td>
                        <td id="T_aab55_row11_col4" class="data row11 col4" >0</td>
                        <td id="T_aab55_row11_col5" class="data row11 col5" >0</td>
                        <td id="T_aab55_row11_col6" class="data row11 col6" >0</td>
                        <td id="T_aab55_row11_col7" class="data row11 col7" >0</td>
                        <td id="T_aab55_row11_col8" class="data row11 col8" >0</td>
                        <td id="T_aab55_row11_col9" class="data row11 col9" >0</td>
                        <td id="T_aab55_row11_col10" class="data row11 col10" >0</td>
                        <td id="T_aab55_row11_col11" class="data row11 col11" >139</td>
                        <td id="T_aab55_row11_col12" class="data row11 col12" >253</td>
                        <td id="T_aab55_row11_col13" class="data row11 col13" >190</td>
                        <td id="T_aab55_row11_col14" class="data row11 col14" >2</td>
                        <td id="T_aab55_row11_col15" class="data row11 col15" >0</td>
                        <td id="T_aab55_row11_col16" class="data row11 col16" >0</td>
                        <td id="T_aab55_row11_col17" class="data row11 col17" >0</td>
                        <td id="T_aab55_row11_col18" class="data row11 col18" >0</td>
                        <td id="T_aab55_row11_col19" class="data row11 col19" >0</td>
                        <td id="T_aab55_row11_col20" class="data row11 col20" >0</td>
                        <td id="T_aab55_row11_col21" class="data row11 col21" >0</td>
                        <td id="T_aab55_row11_col22" class="data row11 col22" >0</td>
                        <td id="T_aab55_row11_col23" class="data row11 col23" >0</td>
                        <td id="T_aab55_row11_col24" class="data row11 col24" >0</td>
                        <td id="T_aab55_row11_col25" class="data row11 col25" >0</td>
                        <td id="T_aab55_row11_col26" class="data row11 col26" >0</td>
                        <td id="T_aab55_row11_col27" class="data row11 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_aab55_row12_col0" class="data row12 col0" >0</td>
                        <td id="T_aab55_row12_col1" class="data row12 col1" >0</td>
                        <td id="T_aab55_row12_col2" class="data row12 col2" >0</td>
                        <td id="T_aab55_row12_col3" class="data row12 col3" >0</td>
                        <td id="T_aab55_row12_col4" class="data row12 col4" >0</td>
                        <td id="T_aab55_row12_col5" class="data row12 col5" >0</td>
                        <td id="T_aab55_row12_col6" class="data row12 col6" >0</td>
                        <td id="T_aab55_row12_col7" class="data row12 col7" >0</td>
                        <td id="T_aab55_row12_col8" class="data row12 col8" >0</td>
                        <td id="T_aab55_row12_col9" class="data row12 col9" >0</td>
                        <td id="T_aab55_row12_col10" class="data row12 col10" >0</td>
                        <td id="T_aab55_row12_col11" class="data row12 col11" >11</td>
                        <td id="T_aab55_row12_col12" class="data row12 col12" >190</td>
                        <td id="T_aab55_row12_col13" class="data row12 col13" >253</td>
                        <td id="T_aab55_row12_col14" class="data row12 col14" >70</td>
                        <td id="T_aab55_row12_col15" class="data row12 col15" >0</td>
                        <td id="T_aab55_row12_col16" class="data row12 col16" >0</td>
                        <td id="T_aab55_row12_col17" class="data row12 col17" >0</td>
                        <td id="T_aab55_row12_col18" class="data row12 col18" >0</td>
                        <td id="T_aab55_row12_col19" class="data row12 col19" >0</td>
                        <td id="T_aab55_row12_col20" class="data row12 col20" >0</td>
                        <td id="T_aab55_row12_col21" class="data row12 col21" >0</td>
                        <td id="T_aab55_row12_col22" class="data row12 col22" >0</td>
                        <td id="T_aab55_row12_col23" class="data row12 col23" >0</td>
                        <td id="T_aab55_row12_col24" class="data row12 col24" >0</td>
                        <td id="T_aab55_row12_col25" class="data row12 col25" >0</td>
                        <td id="T_aab55_row12_col26" class="data row12 col26" >0</td>
                        <td id="T_aab55_row12_col27" class="data row12 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_aab55_row13_col0" class="data row13 col0" >0</td>
                        <td id="T_aab55_row13_col1" class="data row13 col1" >0</td>
                        <td id="T_aab55_row13_col2" class="data row13 col2" >0</td>
                        <td id="T_aab55_row13_col3" class="data row13 col3" >0</td>
                        <td id="T_aab55_row13_col4" class="data row13 col4" >0</td>
                        <td id="T_aab55_row13_col5" class="data row13 col5" >0</td>
                        <td id="T_aab55_row13_col6" class="data row13 col6" >0</td>
                        <td id="T_aab55_row13_col7" class="data row13 col7" >0</td>
                        <td id="T_aab55_row13_col8" class="data row13 col8" >0</td>
                        <td id="T_aab55_row13_col9" class="data row13 col9" >0</td>
                        <td id="T_aab55_row13_col10" class="data row13 col10" >0</td>
                        <td id="T_aab55_row13_col11" class="data row13 col11" >0</td>
                        <td id="T_aab55_row13_col12" class="data row13 col12" >35</td>
                        <td id="T_aab55_row13_col13" class="data row13 col13" >241</td>
                        <td id="T_aab55_row13_col14" class="data row13 col14" >225</td>
                        <td id="T_aab55_row13_col15" class="data row13 col15" >160</td>
                        <td id="T_aab55_row13_col16" class="data row13 col16" >108</td>
                        <td id="T_aab55_row13_col17" class="data row13 col17" >1</td>
                        <td id="T_aab55_row13_col18" class="data row13 col18" >0</td>
                        <td id="T_aab55_row13_col19" class="data row13 col19" >0</td>
                        <td id="T_aab55_row13_col20" class="data row13 col20" >0</td>
                        <td id="T_aab55_row13_col21" class="data row13 col21" >0</td>
                        <td id="T_aab55_row13_col22" class="data row13 col22" >0</td>
                        <td id="T_aab55_row13_col23" class="data row13 col23" >0</td>
                        <td id="T_aab55_row13_col24" class="data row13 col24" >0</td>
                        <td id="T_aab55_row13_col25" class="data row13 col25" >0</td>
                        <td id="T_aab55_row13_col26" class="data row13 col26" >0</td>
                        <td id="T_aab55_row13_col27" class="data row13 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_aab55_row14_col0" class="data row14 col0" >0</td>
                        <td id="T_aab55_row14_col1" class="data row14 col1" >0</td>
                        <td id="T_aab55_row14_col2" class="data row14 col2" >0</td>
                        <td id="T_aab55_row14_col3" class="data row14 col3" >0</td>
                        <td id="T_aab55_row14_col4" class="data row14 col4" >0</td>
                        <td id="T_aab55_row14_col5" class="data row14 col5" >0</td>
                        <td id="T_aab55_row14_col6" class="data row14 col6" >0</td>
                        <td id="T_aab55_row14_col7" class="data row14 col7" >0</td>
                        <td id="T_aab55_row14_col8" class="data row14 col8" >0</td>
                        <td id="T_aab55_row14_col9" class="data row14 col9" >0</td>
                        <td id="T_aab55_row14_col10" class="data row14 col10" >0</td>
                        <td id="T_aab55_row14_col11" class="data row14 col11" >0</td>
                        <td id="T_aab55_row14_col12" class="data row14 col12" >0</td>
                        <td id="T_aab55_row14_col13" class="data row14 col13" >81</td>
                        <td id="T_aab55_row14_col14" class="data row14 col14" >240</td>
                        <td id="T_aab55_row14_col15" class="data row14 col15" >253</td>
                        <td id="T_aab55_row14_col16" class="data row14 col16" >253</td>
                        <td id="T_aab55_row14_col17" class="data row14 col17" >119</td>
                        <td id="T_aab55_row14_col18" class="data row14 col18" >25</td>
                        <td id="T_aab55_row14_col19" class="data row14 col19" >0</td>
                        <td id="T_aab55_row14_col20" class="data row14 col20" >0</td>
                        <td id="T_aab55_row14_col21" class="data row14 col21" >0</td>
                        <td id="T_aab55_row14_col22" class="data row14 col22" >0</td>
                        <td id="T_aab55_row14_col23" class="data row14 col23" >0</td>
                        <td id="T_aab55_row14_col24" class="data row14 col24" >0</td>
                        <td id="T_aab55_row14_col25" class="data row14 col25" >0</td>
                        <td id="T_aab55_row14_col26" class="data row14 col26" >0</td>
                        <td id="T_aab55_row14_col27" class="data row14 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_aab55_row15_col0" class="data row15 col0" >0</td>
                        <td id="T_aab55_row15_col1" class="data row15 col1" >0</td>
                        <td id="T_aab55_row15_col2" class="data row15 col2" >0</td>
                        <td id="T_aab55_row15_col3" class="data row15 col3" >0</td>
                        <td id="T_aab55_row15_col4" class="data row15 col4" >0</td>
                        <td id="T_aab55_row15_col5" class="data row15 col5" >0</td>
                        <td id="T_aab55_row15_col6" class="data row15 col6" >0</td>
                        <td id="T_aab55_row15_col7" class="data row15 col7" >0</td>
                        <td id="T_aab55_row15_col8" class="data row15 col8" >0</td>
                        <td id="T_aab55_row15_col9" class="data row15 col9" >0</td>
                        <td id="T_aab55_row15_col10" class="data row15 col10" >0</td>
                        <td id="T_aab55_row15_col11" class="data row15 col11" >0</td>
                        <td id="T_aab55_row15_col12" class="data row15 col12" >0</td>
                        <td id="T_aab55_row15_col13" class="data row15 col13" >0</td>
                        <td id="T_aab55_row15_col14" class="data row15 col14" >45</td>
                        <td id="T_aab55_row15_col15" class="data row15 col15" >186</td>
                        <td id="T_aab55_row15_col16" class="data row15 col16" >253</td>
                        <td id="T_aab55_row15_col17" class="data row15 col17" >253</td>
                        <td id="T_aab55_row15_col18" class="data row15 col18" >150</td>
                        <td id="T_aab55_row15_col19" class="data row15 col19" >27</td>
                        <td id="T_aab55_row15_col20" class="data row15 col20" >0</td>
                        <td id="T_aab55_row15_col21" class="data row15 col21" >0</td>
                        <td id="T_aab55_row15_col22" class="data row15 col22" >0</td>
                        <td id="T_aab55_row15_col23" class="data row15 col23" >0</td>
                        <td id="T_aab55_row15_col24" class="data row15 col24" >0</td>
                        <td id="T_aab55_row15_col25" class="data row15 col25" >0</td>
                        <td id="T_aab55_row15_col26" class="data row15 col26" >0</td>
                        <td id="T_aab55_row15_col27" class="data row15 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_aab55_row16_col0" class="data row16 col0" >0</td>
                        <td id="T_aab55_row16_col1" class="data row16 col1" >0</td>
                        <td id="T_aab55_row16_col2" class="data row16 col2" >0</td>
                        <td id="T_aab55_row16_col3" class="data row16 col3" >0</td>
                        <td id="T_aab55_row16_col4" class="data row16 col4" >0</td>
                        <td id="T_aab55_row16_col5" class="data row16 col5" >0</td>
                        <td id="T_aab55_row16_col6" class="data row16 col6" >0</td>
                        <td id="T_aab55_row16_col7" class="data row16 col7" >0</td>
                        <td id="T_aab55_row16_col8" class="data row16 col8" >0</td>
                        <td id="T_aab55_row16_col9" class="data row16 col9" >0</td>
                        <td id="T_aab55_row16_col10" class="data row16 col10" >0</td>
                        <td id="T_aab55_row16_col11" class="data row16 col11" >0</td>
                        <td id="T_aab55_row16_col12" class="data row16 col12" >0</td>
                        <td id="T_aab55_row16_col13" class="data row16 col13" >0</td>
                        <td id="T_aab55_row16_col14" class="data row16 col14" >0</td>
                        <td id="T_aab55_row16_col15" class="data row16 col15" >16</td>
                        <td id="T_aab55_row16_col16" class="data row16 col16" >93</td>
                        <td id="T_aab55_row16_col17" class="data row16 col17" >252</td>
                        <td id="T_aab55_row16_col18" class="data row16 col18" >253</td>
                        <td id="T_aab55_row16_col19" class="data row16 col19" >187</td>
                        <td id="T_aab55_row16_col20" class="data row16 col20" >0</td>
                        <td id="T_aab55_row16_col21" class="data row16 col21" >0</td>
                        <td id="T_aab55_row16_col22" class="data row16 col22" >0</td>
                        <td id="T_aab55_row16_col23" class="data row16 col23" >0</td>
                        <td id="T_aab55_row16_col24" class="data row16 col24" >0</td>
                        <td id="T_aab55_row16_col25" class="data row16 col25" >0</td>
                        <td id="T_aab55_row16_col26" class="data row16 col26" >0</td>
                        <td id="T_aab55_row16_col27" class="data row16 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_aab55_row17_col0" class="data row17 col0" >0</td>
                        <td id="T_aab55_row17_col1" class="data row17 col1" >0</td>
                        <td id="T_aab55_row17_col2" class="data row17 col2" >0</td>
                        <td id="T_aab55_row17_col3" class="data row17 col3" >0</td>
                        <td id="T_aab55_row17_col4" class="data row17 col4" >0</td>
                        <td id="T_aab55_row17_col5" class="data row17 col5" >0</td>
                        <td id="T_aab55_row17_col6" class="data row17 col6" >0</td>
                        <td id="T_aab55_row17_col7" class="data row17 col7" >0</td>
                        <td id="T_aab55_row17_col8" class="data row17 col8" >0</td>
                        <td id="T_aab55_row17_col9" class="data row17 col9" >0</td>
                        <td id="T_aab55_row17_col10" class="data row17 col10" >0</td>
                        <td id="T_aab55_row17_col11" class="data row17 col11" >0</td>
                        <td id="T_aab55_row17_col12" class="data row17 col12" >0</td>
                        <td id="T_aab55_row17_col13" class="data row17 col13" >0</td>
                        <td id="T_aab55_row17_col14" class="data row17 col14" >0</td>
                        <td id="T_aab55_row17_col15" class="data row17 col15" >0</td>
                        <td id="T_aab55_row17_col16" class="data row17 col16" >0</td>
                        <td id="T_aab55_row17_col17" class="data row17 col17" >249</td>
                        <td id="T_aab55_row17_col18" class="data row17 col18" >253</td>
                        <td id="T_aab55_row17_col19" class="data row17 col19" >249</td>
                        <td id="T_aab55_row17_col20" class="data row17 col20" >64</td>
                        <td id="T_aab55_row17_col21" class="data row17 col21" >0</td>
                        <td id="T_aab55_row17_col22" class="data row17 col22" >0</td>
                        <td id="T_aab55_row17_col23" class="data row17 col23" >0</td>
                        <td id="T_aab55_row17_col24" class="data row17 col24" >0</td>
                        <td id="T_aab55_row17_col25" class="data row17 col25" >0</td>
                        <td id="T_aab55_row17_col26" class="data row17 col26" >0</td>
                        <td id="T_aab55_row17_col27" class="data row17 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_aab55_row18_col0" class="data row18 col0" >0</td>
                        <td id="T_aab55_row18_col1" class="data row18 col1" >0</td>
                        <td id="T_aab55_row18_col2" class="data row18 col2" >0</td>
                        <td id="T_aab55_row18_col3" class="data row18 col3" >0</td>
                        <td id="T_aab55_row18_col4" class="data row18 col4" >0</td>
                        <td id="T_aab55_row18_col5" class="data row18 col5" >0</td>
                        <td id="T_aab55_row18_col6" class="data row18 col6" >0</td>
                        <td id="T_aab55_row18_col7" class="data row18 col7" >0</td>
                        <td id="T_aab55_row18_col8" class="data row18 col8" >0</td>
                        <td id="T_aab55_row18_col9" class="data row18 col9" >0</td>
                        <td id="T_aab55_row18_col10" class="data row18 col10" >0</td>
                        <td id="T_aab55_row18_col11" class="data row18 col11" >0</td>
                        <td id="T_aab55_row18_col12" class="data row18 col12" >0</td>
                        <td id="T_aab55_row18_col13" class="data row18 col13" >0</td>
                        <td id="T_aab55_row18_col14" class="data row18 col14" >46</td>
                        <td id="T_aab55_row18_col15" class="data row18 col15" >130</td>
                        <td id="T_aab55_row18_col16" class="data row18 col16" >183</td>
                        <td id="T_aab55_row18_col17" class="data row18 col17" >253</td>
                        <td id="T_aab55_row18_col18" class="data row18 col18" >253</td>
                        <td id="T_aab55_row18_col19" class="data row18 col19" >207</td>
                        <td id="T_aab55_row18_col20" class="data row18 col20" >2</td>
                        <td id="T_aab55_row18_col21" class="data row18 col21" >0</td>
                        <td id="T_aab55_row18_col22" class="data row18 col22" >0</td>
                        <td id="T_aab55_row18_col23" class="data row18 col23" >0</td>
                        <td id="T_aab55_row18_col24" class="data row18 col24" >0</td>
                        <td id="T_aab55_row18_col25" class="data row18 col25" >0</td>
                        <td id="T_aab55_row18_col26" class="data row18 col26" >0</td>
                        <td id="T_aab55_row18_col27" class="data row18 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_aab55_row19_col0" class="data row19 col0" >0</td>
                        <td id="T_aab55_row19_col1" class="data row19 col1" >0</td>
                        <td id="T_aab55_row19_col2" class="data row19 col2" >0</td>
                        <td id="T_aab55_row19_col3" class="data row19 col3" >0</td>
                        <td id="T_aab55_row19_col4" class="data row19 col4" >0</td>
                        <td id="T_aab55_row19_col5" class="data row19 col5" >0</td>
                        <td id="T_aab55_row19_col6" class="data row19 col6" >0</td>
                        <td id="T_aab55_row19_col7" class="data row19 col7" >0</td>
                        <td id="T_aab55_row19_col8" class="data row19 col8" >0</td>
                        <td id="T_aab55_row19_col9" class="data row19 col9" >0</td>
                        <td id="T_aab55_row19_col10" class="data row19 col10" >0</td>
                        <td id="T_aab55_row19_col11" class="data row19 col11" >0</td>
                        <td id="T_aab55_row19_col12" class="data row19 col12" >39</td>
                        <td id="T_aab55_row19_col13" class="data row19 col13" >148</td>
                        <td id="T_aab55_row19_col14" class="data row19 col14" >229</td>
                        <td id="T_aab55_row19_col15" class="data row19 col15" >253</td>
                        <td id="T_aab55_row19_col16" class="data row19 col16" >253</td>
                        <td id="T_aab55_row19_col17" class="data row19 col17" >253</td>
                        <td id="T_aab55_row19_col18" class="data row19 col18" >250</td>
                        <td id="T_aab55_row19_col19" class="data row19 col19" >182</td>
                        <td id="T_aab55_row19_col20" class="data row19 col20" >0</td>
                        <td id="T_aab55_row19_col21" class="data row19 col21" >0</td>
                        <td id="T_aab55_row19_col22" class="data row19 col22" >0</td>
                        <td id="T_aab55_row19_col23" class="data row19 col23" >0</td>
                        <td id="T_aab55_row19_col24" class="data row19 col24" >0</td>
                        <td id="T_aab55_row19_col25" class="data row19 col25" >0</td>
                        <td id="T_aab55_row19_col26" class="data row19 col26" >0</td>
                        <td id="T_aab55_row19_col27" class="data row19 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_aab55_row20_col0" class="data row20 col0" >0</td>
                        <td id="T_aab55_row20_col1" class="data row20 col1" >0</td>
                        <td id="T_aab55_row20_col2" class="data row20 col2" >0</td>
                        <td id="T_aab55_row20_col3" class="data row20 col3" >0</td>
                        <td id="T_aab55_row20_col4" class="data row20 col4" >0</td>
                        <td id="T_aab55_row20_col5" class="data row20 col5" >0</td>
                        <td id="T_aab55_row20_col6" class="data row20 col6" >0</td>
                        <td id="T_aab55_row20_col7" class="data row20 col7" >0</td>
                        <td id="T_aab55_row20_col8" class="data row20 col8" >0</td>
                        <td id="T_aab55_row20_col9" class="data row20 col9" >0</td>
                        <td id="T_aab55_row20_col10" class="data row20 col10" >24</td>
                        <td id="T_aab55_row20_col11" class="data row20 col11" >114</td>
                        <td id="T_aab55_row20_col12" class="data row20 col12" >221</td>
                        <td id="T_aab55_row20_col13" class="data row20 col13" >253</td>
                        <td id="T_aab55_row20_col14" class="data row20 col14" >253</td>
                        <td id="T_aab55_row20_col15" class="data row20 col15" >253</td>
                        <td id="T_aab55_row20_col16" class="data row20 col16" >253</td>
                        <td id="T_aab55_row20_col17" class="data row20 col17" >201</td>
                        <td id="T_aab55_row20_col18" class="data row20 col18" >78</td>
                        <td id="T_aab55_row20_col19" class="data row20 col19" >0</td>
                        <td id="T_aab55_row20_col20" class="data row20 col20" >0</td>
                        <td id="T_aab55_row20_col21" class="data row20 col21" >0</td>
                        <td id="T_aab55_row20_col22" class="data row20 col22" >0</td>
                        <td id="T_aab55_row20_col23" class="data row20 col23" >0</td>
                        <td id="T_aab55_row20_col24" class="data row20 col24" >0</td>
                        <td id="T_aab55_row20_col25" class="data row20 col25" >0</td>
                        <td id="T_aab55_row20_col26" class="data row20 col26" >0</td>
                        <td id="T_aab55_row20_col27" class="data row20 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_aab55_row21_col0" class="data row21 col0" >0</td>
                        <td id="T_aab55_row21_col1" class="data row21 col1" >0</td>
                        <td id="T_aab55_row21_col2" class="data row21 col2" >0</td>
                        <td id="T_aab55_row21_col3" class="data row21 col3" >0</td>
                        <td id="T_aab55_row21_col4" class="data row21 col4" >0</td>
                        <td id="T_aab55_row21_col5" class="data row21 col5" >0</td>
                        <td id="T_aab55_row21_col6" class="data row21 col6" >0</td>
                        <td id="T_aab55_row21_col7" class="data row21 col7" >0</td>
                        <td id="T_aab55_row21_col8" class="data row21 col8" >23</td>
                        <td id="T_aab55_row21_col9" class="data row21 col9" >66</td>
                        <td id="T_aab55_row21_col10" class="data row21 col10" >213</td>
                        <td id="T_aab55_row21_col11" class="data row21 col11" >253</td>
                        <td id="T_aab55_row21_col12" class="data row21 col12" >253</td>
                        <td id="T_aab55_row21_col13" class="data row21 col13" >253</td>
                        <td id="T_aab55_row21_col14" class="data row21 col14" >253</td>
                        <td id="T_aab55_row21_col15" class="data row21 col15" >198</td>
                        <td id="T_aab55_row21_col16" class="data row21 col16" >81</td>
                        <td id="T_aab55_row21_col17" class="data row21 col17" >2</td>
                        <td id="T_aab55_row21_col18" class="data row21 col18" >0</td>
                        <td id="T_aab55_row21_col19" class="data row21 col19" >0</td>
                        <td id="T_aab55_row21_col20" class="data row21 col20" >0</td>
                        <td id="T_aab55_row21_col21" class="data row21 col21" >0</td>
                        <td id="T_aab55_row21_col22" class="data row21 col22" >0</td>
                        <td id="T_aab55_row21_col23" class="data row21 col23" >0</td>
                        <td id="T_aab55_row21_col24" class="data row21 col24" >0</td>
                        <td id="T_aab55_row21_col25" class="data row21 col25" >0</td>
                        <td id="T_aab55_row21_col26" class="data row21 col26" >0</td>
                        <td id="T_aab55_row21_col27" class="data row21 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_aab55_row22_col0" class="data row22 col0" >0</td>
                        <td id="T_aab55_row22_col1" class="data row22 col1" >0</td>
                        <td id="T_aab55_row22_col2" class="data row22 col2" >0</td>
                        <td id="T_aab55_row22_col3" class="data row22 col3" >0</td>
                        <td id="T_aab55_row22_col4" class="data row22 col4" >0</td>
                        <td id="T_aab55_row22_col5" class="data row22 col5" >0</td>
                        <td id="T_aab55_row22_col6" class="data row22 col6" >18</td>
                        <td id="T_aab55_row22_col7" class="data row22 col7" >171</td>
                        <td id="T_aab55_row22_col8" class="data row22 col8" >219</td>
                        <td id="T_aab55_row22_col9" class="data row22 col9" >253</td>
                        <td id="T_aab55_row22_col10" class="data row22 col10" >253</td>
                        <td id="T_aab55_row22_col11" class="data row22 col11" >253</td>
                        <td id="T_aab55_row22_col12" class="data row22 col12" >253</td>
                        <td id="T_aab55_row22_col13" class="data row22 col13" >195</td>
                        <td id="T_aab55_row22_col14" class="data row22 col14" >80</td>
                        <td id="T_aab55_row22_col15" class="data row22 col15" >9</td>
                        <td id="T_aab55_row22_col16" class="data row22 col16" >0</td>
                        <td id="T_aab55_row22_col17" class="data row22 col17" >0</td>
                        <td id="T_aab55_row22_col18" class="data row22 col18" >0</td>
                        <td id="T_aab55_row22_col19" class="data row22 col19" >0</td>
                        <td id="T_aab55_row22_col20" class="data row22 col20" >0</td>
                        <td id="T_aab55_row22_col21" class="data row22 col21" >0</td>
                        <td id="T_aab55_row22_col22" class="data row22 col22" >0</td>
                        <td id="T_aab55_row22_col23" class="data row22 col23" >0</td>
                        <td id="T_aab55_row22_col24" class="data row22 col24" >0</td>
                        <td id="T_aab55_row22_col25" class="data row22 col25" >0</td>
                        <td id="T_aab55_row22_col26" class="data row22 col26" >0</td>
                        <td id="T_aab55_row22_col27" class="data row22 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_aab55_row23_col0" class="data row23 col0" >0</td>
                        <td id="T_aab55_row23_col1" class="data row23 col1" >0</td>
                        <td id="T_aab55_row23_col2" class="data row23 col2" >0</td>
                        <td id="T_aab55_row23_col3" class="data row23 col3" >0</td>
                        <td id="T_aab55_row23_col4" class="data row23 col4" >55</td>
                        <td id="T_aab55_row23_col5" class="data row23 col5" >172</td>
                        <td id="T_aab55_row23_col6" class="data row23 col6" >226</td>
                        <td id="T_aab55_row23_col7" class="data row23 col7" >253</td>
                        <td id="T_aab55_row23_col8" class="data row23 col8" >253</td>
                        <td id="T_aab55_row23_col9" class="data row23 col9" >253</td>
                        <td id="T_aab55_row23_col10" class="data row23 col10" >253</td>
                        <td id="T_aab55_row23_col11" class="data row23 col11" >244</td>
                        <td id="T_aab55_row23_col12" class="data row23 col12" >133</td>
                        <td id="T_aab55_row23_col13" class="data row23 col13" >11</td>
                        <td id="T_aab55_row23_col14" class="data row23 col14" >0</td>
                        <td id="T_aab55_row23_col15" class="data row23 col15" >0</td>
                        <td id="T_aab55_row23_col16" class="data row23 col16" >0</td>
                        <td id="T_aab55_row23_col17" class="data row23 col17" >0</td>
                        <td id="T_aab55_row23_col18" class="data row23 col18" >0</td>
                        <td id="T_aab55_row23_col19" class="data row23 col19" >0</td>
                        <td id="T_aab55_row23_col20" class="data row23 col20" >0</td>
                        <td id="T_aab55_row23_col21" class="data row23 col21" >0</td>
                        <td id="T_aab55_row23_col22" class="data row23 col22" >0</td>
                        <td id="T_aab55_row23_col23" class="data row23 col23" >0</td>
                        <td id="T_aab55_row23_col24" class="data row23 col24" >0</td>
                        <td id="T_aab55_row23_col25" class="data row23 col25" >0</td>
                        <td id="T_aab55_row23_col26" class="data row23 col26" >0</td>
                        <td id="T_aab55_row23_col27" class="data row23 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_aab55_row24_col0" class="data row24 col0" >0</td>
                        <td id="T_aab55_row24_col1" class="data row24 col1" >0</td>
                        <td id="T_aab55_row24_col2" class="data row24 col2" >0</td>
                        <td id="T_aab55_row24_col3" class="data row24 col3" >0</td>
                        <td id="T_aab55_row24_col4" class="data row24 col4" >136</td>
                        <td id="T_aab55_row24_col5" class="data row24 col5" >253</td>
                        <td id="T_aab55_row24_col6" class="data row24 col6" >253</td>
                        <td id="T_aab55_row24_col7" class="data row24 col7" >253</td>
                        <td id="T_aab55_row24_col8" class="data row24 col8" >212</td>
                        <td id="T_aab55_row24_col9" class="data row24 col9" >135</td>
                        <td id="T_aab55_row24_col10" class="data row24 col10" >132</td>
                        <td id="T_aab55_row24_col11" class="data row24 col11" >16</td>
                        <td id="T_aab55_row24_col12" class="data row24 col12" >0</td>
                        <td id="T_aab55_row24_col13" class="data row24 col13" >0</td>
                        <td id="T_aab55_row24_col14" class="data row24 col14" >0</td>
                        <td id="T_aab55_row24_col15" class="data row24 col15" >0</td>
                        <td id="T_aab55_row24_col16" class="data row24 col16" >0</td>
                        <td id="T_aab55_row24_col17" class="data row24 col17" >0</td>
                        <td id="T_aab55_row24_col18" class="data row24 col18" >0</td>
                        <td id="T_aab55_row24_col19" class="data row24 col19" >0</td>
                        <td id="T_aab55_row24_col20" class="data row24 col20" >0</td>
                        <td id="T_aab55_row24_col21" class="data row24 col21" >0</td>
                        <td id="T_aab55_row24_col22" class="data row24 col22" >0</td>
                        <td id="T_aab55_row24_col23" class="data row24 col23" >0</td>
                        <td id="T_aab55_row24_col24" class="data row24 col24" >0</td>
                        <td id="T_aab55_row24_col25" class="data row24 col25" >0</td>
                        <td id="T_aab55_row24_col26" class="data row24 col26" >0</td>
                        <td id="T_aab55_row24_col27" class="data row24 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_aab55_row25_col0" class="data row25 col0" >0</td>
                        <td id="T_aab55_row25_col1" class="data row25 col1" >0</td>
                        <td id="T_aab55_row25_col2" class="data row25 col2" >0</td>
                        <td id="T_aab55_row25_col3" class="data row25 col3" >0</td>
                        <td id="T_aab55_row25_col4" class="data row25 col4" >0</td>
                        <td id="T_aab55_row25_col5" class="data row25 col5" >0</td>
                        <td id="T_aab55_row25_col6" class="data row25 col6" >0</td>
                        <td id="T_aab55_row25_col7" class="data row25 col7" >0</td>
                        <td id="T_aab55_row25_col8" class="data row25 col8" >0</td>
                        <td id="T_aab55_row25_col9" class="data row25 col9" >0</td>
                        <td id="T_aab55_row25_col10" class="data row25 col10" >0</td>
                        <td id="T_aab55_row25_col11" class="data row25 col11" >0</td>
                        <td id="T_aab55_row25_col12" class="data row25 col12" >0</td>
                        <td id="T_aab55_row25_col13" class="data row25 col13" >0</td>
                        <td id="T_aab55_row25_col14" class="data row25 col14" >0</td>
                        <td id="T_aab55_row25_col15" class="data row25 col15" >0</td>
                        <td id="T_aab55_row25_col16" class="data row25 col16" >0</td>
                        <td id="T_aab55_row25_col17" class="data row25 col17" >0</td>
                        <td id="T_aab55_row25_col18" class="data row25 col18" >0</td>
                        <td id="T_aab55_row25_col19" class="data row25 col19" >0</td>
                        <td id="T_aab55_row25_col20" class="data row25 col20" >0</td>
                        <td id="T_aab55_row25_col21" class="data row25 col21" >0</td>
                        <td id="T_aab55_row25_col22" class="data row25 col22" >0</td>
                        <td id="T_aab55_row25_col23" class="data row25 col23" >0</td>
                        <td id="T_aab55_row25_col24" class="data row25 col24" >0</td>
                        <td id="T_aab55_row25_col25" class="data row25 col25" >0</td>
                        <td id="T_aab55_row25_col26" class="data row25 col26" >0</td>
                        <td id="T_aab55_row25_col27" class="data row25 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_aab55_row26_col0" class="data row26 col0" >0</td>
                        <td id="T_aab55_row26_col1" class="data row26 col1" >0</td>
                        <td id="T_aab55_row26_col2" class="data row26 col2" >0</td>
                        <td id="T_aab55_row26_col3" class="data row26 col3" >0</td>
                        <td id="T_aab55_row26_col4" class="data row26 col4" >0</td>
                        <td id="T_aab55_row26_col5" class="data row26 col5" >0</td>
                        <td id="T_aab55_row26_col6" class="data row26 col6" >0</td>
                        <td id="T_aab55_row26_col7" class="data row26 col7" >0</td>
                        <td id="T_aab55_row26_col8" class="data row26 col8" >0</td>
                        <td id="T_aab55_row26_col9" class="data row26 col9" >0</td>
                        <td id="T_aab55_row26_col10" class="data row26 col10" >0</td>
                        <td id="T_aab55_row26_col11" class="data row26 col11" >0</td>
                        <td id="T_aab55_row26_col12" class="data row26 col12" >0</td>
                        <td id="T_aab55_row26_col13" class="data row26 col13" >0</td>
                        <td id="T_aab55_row26_col14" class="data row26 col14" >0</td>
                        <td id="T_aab55_row26_col15" class="data row26 col15" >0</td>
                        <td id="T_aab55_row26_col16" class="data row26 col16" >0</td>
                        <td id="T_aab55_row26_col17" class="data row26 col17" >0</td>
                        <td id="T_aab55_row26_col18" class="data row26 col18" >0</td>
                        <td id="T_aab55_row26_col19" class="data row26 col19" >0</td>
                        <td id="T_aab55_row26_col20" class="data row26 col20" >0</td>
                        <td id="T_aab55_row26_col21" class="data row26 col21" >0</td>
                        <td id="T_aab55_row26_col22" class="data row26 col22" >0</td>
                        <td id="T_aab55_row26_col23" class="data row26 col23" >0</td>
                        <td id="T_aab55_row26_col24" class="data row26 col24" >0</td>
                        <td id="T_aab55_row26_col25" class="data row26 col25" >0</td>
                        <td id="T_aab55_row26_col26" class="data row26 col26" >0</td>
                        <td id="T_aab55_row26_col27" class="data row26 col27" >0</td>
            </tr>
            <tr>
                        <th id="T_aab55_level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_aab55_row27_col0" class="data row27 col0" >0</td>
                        <td id="T_aab55_row27_col1" class="data row27 col1" >0</td>
                        <td id="T_aab55_row27_col2" class="data row27 col2" >0</td>
                        <td id="T_aab55_row27_col3" class="data row27 col3" >0</td>
                        <td id="T_aab55_row27_col4" class="data row27 col4" >0</td>
                        <td id="T_aab55_row27_col5" class="data row27 col5" >0</td>
                        <td id="T_aab55_row27_col6" class="data row27 col6" >0</td>
                        <td id="T_aab55_row27_col7" class="data row27 col7" >0</td>
                        <td id="T_aab55_row27_col8" class="data row27 col8" >0</td>
                        <td id="T_aab55_row27_col9" class="data row27 col9" >0</td>
                        <td id="T_aab55_row27_col10" class="data row27 col10" >0</td>
                        <td id="T_aab55_row27_col11" class="data row27 col11" >0</td>
                        <td id="T_aab55_row27_col12" class="data row27 col12" >0</td>
                        <td id="T_aab55_row27_col13" class="data row27 col13" >0</td>
                        <td id="T_aab55_row27_col14" class="data row27 col14" >0</td>
                        <td id="T_aab55_row27_col15" class="data row27 col15" >0</td>
                        <td id="T_aab55_row27_col16" class="data row27 col16" >0</td>
                        <td id="T_aab55_row27_col17" class="data row27 col17" >0</td>
                        <td id="T_aab55_row27_col18" class="data row27 col18" >0</td>
                        <td id="T_aab55_row27_col19" class="data row27 col19" >0</td>
                        <td id="T_aab55_row27_col20" class="data row27 col20" >0</td>
                        <td id="T_aab55_row27_col21" class="data row27 col21" >0</td>
                        <td id="T_aab55_row27_col22" class="data row27 col22" >0</td>
                        <td id="T_aab55_row27_col23" class="data row27 col23" >0</td>
                        <td id="T_aab55_row27_col24" class="data row27 col24" >0</td>
                        <td id="T_aab55_row27_col25" class="data row27 col25" >0</td>
                        <td id="T_aab55_row27_col26" class="data row27 col26" >0</td>
                        <td id="T_aab55_row27_col27" class="data row27 col27" >0</td>
            </tr>
    </tbody></table>



Now in the next operation we will use another PyTorch method that stacks all our tensors on top of each other, creating a new tensor with three dimensions instead of two. So effectively we end up with one 3D tensor instead of a list of two-dimensional tensors.
The torch.stack() method stacks a list of tensors together. As you see when we check the shape of stacked_zeros, for example, we end up with a 5923 by 28 by 28 tensor. Just imagine 5923 2d tensors stacked on top of each other.

```python
stacked_zeros = torch.stack(zeros_tensors).float()/255
stacked_ones = torch.stack(ones_tensors).float()/255
stacked_twos = torch.stack(twos_tensors).float()/255
stacked_threes = torch.stack(threes_tensors).float()/255
stacked_fours = torch.stack(fours_tensors).float()/255
stacked_fives = torch.stack(fives_tensors).float()/255
stacked_sixes = torch.stack(sixes_tensors).float()/255
stacked_sevens = torch.stack(sevens_tensors).float()/255
stacked_eights = torch.stack(eights_tensors).float()/255
stacked_nines = torch.stack(nines_tensors).float()/255
```

```python
stacked_zeros.shape
```




    torch.Size([5923, 28, 28])



Now we are nearly getting the data into the format we want. We will now "concatenate" (hence the torch.cat() method) or string together all the previous tensors giving us one big tensor of 60,000 images, containing all the image tensors.
As you can see we also applied the [view method](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html). This method can change the dimensions of a tensor while keeping all the data. We changed the last two of the dimensions (28 and 28) by one dimension of 784. As the first dimension we passed "-1" to the method, telling pytorch to let that dimension be flexible and adapt to the other dimensions.
If we now look at the shape, we can see that we have 60,000 rows (one for each image) of 784 columns. This means that instead of keeping the 28 by 28 we reduced all the images to one row of 784 digits. We just have one big line of numbers now for each image. For the neural net to train on this data, it doesn't matter that we have the original 28 by 28 dimension.

```python
train_x = torch.cat([stacked_zeros, stacked_ones, stacked_twos, stacked_threes, stacked_fours, stacked_fives, stacked_sixes, stacked_sevens, stacked_eights, stacked_nines]).view(-1, 28*28)
train_x.shape
```




    torch.Size([60000, 784])



Now we also want to add our labels to the dataset. Labels are effectively a digit that we add to a row of the dataset to specify which digit it actually is. This will then be used by the model to train the neural net. The labels will also be used to check the accuracy of the model while doing training, to see how it evolves.
Below we are creating a tensor which we will paste (so to speak) as a last column to the train_x tensor. We know how many lines of each image we have in the dataset, i.e. len(zeros), so we know how many digits we have to add. Remember, all the rows in the train_x tensor are all still nicely in order so we will just add digits from 0 to 9 as a last column to create a complete dataset later on.
We also check the dimension of our train_y tensor, and using the unsueeze() method on the tensor we have given it the right dimension, being 60,000 rows long and 1 coloumn wide. The unsqueeze method has changed the dimension of being one row of 60,000 digits to one column.

```python
train_y = tensor([0]*len(zeros) + [1]*len(ones) + [2]*len(twos) + [3]*len(threes) + [4]*len(fours) + [5]*len(fives) + [6]*len(sixes) + [7]*len(sevens) + [8]*len(eights) + [9]*len(nines)).unsqueeze(1)
```

```python
train_y.shape
```




    torch.Size([60000, 1])



Now let's paste the train_x and the train_y together so we create dset, a dataset of tuples. So that later on we can unpack them.
From the fast.ai course : "A `Dataset` in PyTorch is required to return a tuple of `(x,y)` when indexed. Python provides a `zip` function which, when combined with `list`, provides a simple way to get this functionality.
Now with dset, our dataset (training dataset) is complete. We'll complete the exact same steps to get a validation dataset so that we have a completely separate dataset to check the accuracy of our model later on.

```python
dset = list(zip(train_x, train_y))
```

We can now unpack the first image of the dataset and its label, like below. x is a tensor of 1 by 784, the image of a digit. y is giving the label. In this case it's the number 0, which tells us the image tensor is a zero digit.

```python
x, y = dset[0]
x.shape, y
```




    (torch.Size([784]), tensor([0]))



Now we'll do exactly the same for the validation set. We'll open the images, put them in a list by digit, stack them all up in a tensor and reduce the dimension to have 1 by 784 tensors stacked on top of each other, giving us a 3D tensor.

```python
valid_0_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'0').ls()])
valid_0_tens = valid_0_tens.float()/255
valid_1_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'1').ls()])
valid_1_tens = valid_1_tens.float()/255
valid_2_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'2').ls()])
valid_2_tens = valid_2_tens.float()/255
valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'3').ls()])
valid_3_tens = valid_3_tens.float()/255
valid_4_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'4').ls()])
valid_4_tens = valid_4_tens.float()/255
valid_5_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'5').ls()])
valid_5_tens = valid_5_tens.float()/255
valid_6_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'6').ls()])
valid_6_tens = valid_6_tens.float()/255
valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'7').ls()])
valid_7_tens = valid_7_tens.float()/255
valid_8_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'8').ls()])
valid_8_tens = valid_8_tens.float()/255
valid_9_tens = torch.stack([tensor(Image.open(o)) for o in (path/'testing'/'9').ls()])
valid_9_tens = valid_9_tens.float()/255
```

```python
valid_0_tens.shape, valid_2_tens.shape
```




    (torch.Size([980, 28, 28]), torch.Size([1032, 28, 28]))



```python
valid_x = torch.cat([valid_0_tens, valid_1_tens, valid_2_tens, valid_3_tens, valid_4_tens, valid_5_tens, valid_6_tens, valid_7_tens, valid_8_tens, valid_9_tens]).view(-1, 28*28)
valid_y = tensor([0]*len(valid_0_tens) + [1]*len(valid_1_tens) + [2]*len(valid_2_tens) + [3]*len(valid_3_tens) + [4]*len(valid_4_tens) + [5]*len(valid_5_tens) + [6]*len(valid_6_tens) + [7]*len(valid_7_tens) + [8]*len(valid_8_tens) + [9]*len(valid_9_tens)).unsqueeze(1)
valid_dset = list(zip(valid_x,valid_y))
```

This finally gives us our validation dataset (valid_dset)

```python
x, y = valid_dset[7000]
x.shape, y
```




    (torch.Size([784]), tensor([7]))



Now that we have all our data the way we want, we can look at building up our simple neural net. In order to do that we need to take a step back and think about what a neural net and deep learning actually is.

For context, we can specify that a deep learning is a part of machine learning. 
In machine learning we don't program the computer with exact instructions. What we do instead is: we show the computer examples and then there is an automatic means of improving it's performance.

Imagine how you would "program" a computer to recognize a dog or a cat. It would be very difficult to give it exact instructions on that. These kinds of image classification tasks our ideal for our neural net.

How does it actually work? We need the following steps:

- we need the idea of a weight assignment
- every weight assignment has an actual performance
- there needs to be an automatic way to test this performance
- we need a mechanism to change the weight assignments in order to improve the performance

How does this tie up to a neural net. We are going to give each of the 784 pixels a weight assignment. In other words we create 784 random numbers to go with each pixel. Then we will multiply the weights with the respective pixels, effectively doing a matrix multiplication. After doing that we add a bias to it. 

If x is the pixel and w is the weight, we get this formula : result = x times w plus b 
In python the "@" sign is used for matrix multiplication. We multiply every image with our weights and add the bias.

As you see in the "simple_net" function below, we have two "layers" where we do this matrix multiplication. In between that we add a ReLu function. (Rectified Linear Unit). This sounds very complicated but all it does is this: If the number is below zero, it will make it zero. If it's bigger than zero, the number stays the same. 
Why do we do that? If you remember, "x times w plus b" is a linear function. If we would just add one linear function after the other, it could be described as a new linear function. (with different parameters) 
By adding the ReLu function in between we make the function so that it is able to become much more flexible than a simple linear function, making our model a lot better.

For our example, we will start with the following parameters: w1 and b1 (weights and bias for our first layer) and w2 and b2 (weights and bias for the second layer). Finally we end up with 10 numbers or results, which we call the "activations" in machine learning jargon. These 10 activations will for each image give a prediction of how likely it is that the image is a certain digit.

After the last layer, we use the log_softmax() algorithm in PyTorch, which we will get into later on.

Let's write the function that can create our random initial parameters. It takes the size or dimension.

```python
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
```

Since we use two layers in our simple_net model, we need to make sure that we have enough numbers (30) coming out of the first layer to take as input on the second layer. The last layer ends with 10 numbers or activations. These will become the predictions for a certain image being a certain number

```python
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,10))
b2 = init_params(10)
```

In this simple but powerful function we create our neural net and specify the layers. We are only using two layers here but the amount of layers we can use is endless. The more layers we use the more performant our model can become!

```python
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    res = torch.log_softmax(res, dim=1)
    return res
```

By using the fast.ai Dataloader function, we can create batches. This way we can train our model in batches instead of using the 60,000 images at once. So we take a batch of 12 images in this example. We calculate the activations, we update the weights and then we start the process again.
As you see below the xb variable gives us 12 rows of 784 pixel images. The yb variable gives us 12 labels that specify which digit is is.


```python
dl = DataLoader(dset, batch_size=12, shuffle=True)
xb,yb = first(dl)
xb.shape,yb.shape
```




    (torch.Size([12, 784]), torch.Size([12, 1]))



Let's create a dataloader for the validations set too, so we can print out the accuracy each time we go through the whole dataset. (also called an "epoch")

```python
valid_dl = DataLoader(valid_dset, batch_size=12, shuffle=True)
```

```python
xb, yb = first(dl)
xb.shape, yb.shape
```




    (torch.Size([12, 784]), torch.Size([12, 1]))



Let's start with putting the first batch of 12 images through our model. Remember, the weights are random so we will get pretty random results. We get a tensor of 12 rows with 10 predictions each. 

Remember in the last layer we took the results of our calculations and use the PyTorch method called log_softmax(). This did two things:

- take the softmax of the numbers: this takes the activations (i.e. results of our calculations) and makes them into probabilities which all add up to 1. We will see later on that when we change the weights to optimise our model, the goal of it is to make a correct prediction show a number close to 1 and a wrong prediction close to 0.

- it also takes the log. For numbers between 0 and 1, the log gives us numbers between minus infinity and zero. So when you see a zero in the list of predictions below, it means the model "thinks" that digit is the correct prediction. Why use the log? We will see that when we change the weights using our loss function, it's easier for the computer when looking at these numbers because the differences are bigger.

```python
preds= simple_net(xb)
preds
```




    tensor([[-8.3995e+01, -8.8755e+01, -5.3084e+01, -7.2887e+01, -5.5825e+01, -6.1979e+01, -5.7795e+01, -3.4197e+01, -9.5367e-07, -1.3850e+01],
            [-5.9557e+01, -1.1856e+02, -3.5212e+01, -1.0678e+02, -6.6234e+01, -8.4050e+01, -1.3987e+02, -7.4412e+01,  0.0000e+00, -5.2639e+01],
            [-6.7138e+01, -1.4267e+02, -3.9344e+01, -7.0194e+01, -9.3688e+01, -1.5169e+02, -6.1433e+01, -3.0317e+01,  0.0000e+00, -2.9096e+01],
            [-1.6496e+02, -1.6859e+02, -1.3029e+02, -8.5140e+01, -1.4555e+02, -1.4491e+02, -1.1815e+02, -1.3522e+02, -8.5044e+01,  0.0000e+00],
            [-3.0546e+01, -5.7355e+01, -6.5549e+01, -5.2643e+01, -7.4359e+01, -3.1528e+01, -5.7887e+01, -3.6325e+01, -1.1228e+01, -1.3351e-05],
            [-1.1843e+02, -1.1305e+02, -4.6417e+01, -6.8255e+01, -9.7619e+01, -1.0997e+02, -1.0039e+02, -9.2649e+01, -8.6147e+00, -1.8142e-04],
            [-6.8548e+01, -6.8306e+01, -4.2830e+01, -1.1437e+02, -9.6202e+01, -5.3297e+01, -9.6925e+01, -7.9097e+01, -2.0028e+01,  0.0000e+00],
            [-1.1872e+02, -7.3489e+01, -9.9053e+01, -3.1166e+01, -2.2211e+01, -4.9665e+01, -9.7209e+01, -9.6563e+01, -7.1856e+01,  0.0000e+00],
            [-8.1209e+01, -8.6876e+01, -9.5427e+01, -7.2390e+01, -5.8609e+01, -5.7028e+01, -1.1787e+02, -8.2441e+01, -7.4513e+00, -5.8086e-04],
            [-7.8678e+01, -8.2223e+01, -3.7264e+01, -8.4110e+01, -6.6882e+01, -4.4760e+01, -6.6715e+01, -7.7473e+01, -3.4932e+01,  0.0000e+00],
            [-1.2547e+02, -1.7304e+02, -9.0148e+01, -1.0656e+02, -7.3735e+01, -1.6565e+02, -8.3019e+01, -9.0835e+01, -7.6103e+01,  0.0000e+00],
            [-8.7988e+01, -7.0126e+01, -6.3045e+01, -4.8035e+01, -3.0917e+01, -2.8748e+01, -7.0165e+01, -7.2555e+01, -6.6636e+01,  0.0000e+00]], grad_fn=<LogSoftmaxBackward>)



Now of course we want to see how our model is doing. There are two ways to "see" how the model is doing. One is the way computers look at it, using the loss function. 
The other way is how we can see how to model is doing as humans: accuracy. (for example: 10% of predictions is correct when we go through the whole dataset)

We have looked at the predictions for the first 12 images. Now let's look at the labels for those same images. The prediction for the first image is the first number, and so on...

```python
yb
```




    tensor([[8],
            [9],
            [2],
            [8],
            [7],
            [8],
            [3],
            [4],
            [9],
            [3],
            [9],
            [1]])



Now we need to find a way to link this to our predictions that we got earlier. Each image has 10 predictions. The first prediction is the prediction for that image being a zero, the second digit is the prediction for that digit being a one, and so on.

We need to define a loss function by which we can "automatically" update the weights so that next time it performs a little bit better. In other words the loss needs to become smaller and smaller by updating our weights in the correct direction.

So here is what we do: we give the predictions tensor and our labels tensor to the nll_loss() function. It does this:

- for each image it takes only the activation related to the correct digit, by using the label number
- it ignores all the other predictions for that image
- it gives us the loss (i.e. the difference between the prediction and zero)

We know that the model should predict a zero for this number. (the log of 1 is zero and the label indicates what is the correct number). Now we need to find a way to make sure these predictions end up closer to that zero. We do this using gradients. Gradients tell us in which direction we need to change the weights in order to make the loss smaller.

Here we see the losses for the first 12 images: (we use the squeeze() method because nll_loss() wants the tensor in a certain dimension.

```python
F.nll_loss(preds, yb.squeeze(), reduction='none')
```




    tensor([9.5367e-07, 5.2639e+01, 3.9344e+01, 8.5044e+01, 3.6325e+01, 8.6147e+00, 1.1437e+02, 2.2211e+01, 5.8086e-04, 8.4110e+01, -0.0000e+00, 7.0126e+01], grad_fn=<NllLossBackward>)



Let's define the loss function. This function will be used later by our model to improve its performance

```python
def my_mnist_loss(predictions, targets):
    return F.nll_loss(predictions, targets.squeeze())
```

Finally let's calculate the actual loss for our first batch of 12:

```python
loss = my_mnist_loss(preds, yb)
loss
```




    tensor(42.7321, grad_fn=<NllLossBackward>)



What we built up ourselves from scratch, is actually available in PyTorch as "cross entropy". So we could define our loss function as nn.CrossEntropyLoss() and achieve exactly the same result. 

```python
loss_func = nn.CrossEntropyLoss()
loss_func(preds, yb.squeeze())
```




    tensor(42.7321, grad_fn=<NllLossBackward>)



Now that we found a way to show the model how far off it is, let's find a way to show actual humans how well the model is doing. We want to go trough the predictions for the batch of 12 and for each image (i.e. each row) we want it to pick the highest number, because effectively that will be closest to zero, meaning that that is the number the model picks as the winner. 
We can do that using the argmax() method.

```python
preds.argmax(dim=1)
```




    tensor([8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9])



Now let's compare that to what the labels say is true for these images:

```python
yb.squeeze()
```




    tensor([8, 9, 2, 8, 7, 8, 3, 4, 9, 3, 9, 1])



With this function we will be able to find out the accuracy for a batch. It take the two variables we discussed above and sees how many times it's correct

```python
def batch_accuracy(preds, yb):
    return (preds.argmax(dim=1) == yb.squeeze()).float().mean()
```

```python
batch_accuracy(simple_net(xb), yb)
```




    tensor(0.2500)



Now we don't want to see the accuracy for one batch, but for the whole dataset:

```python
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
```

Your results may differ, but the number should be around 10%. That makes sense because the first time around, we are using random weights and we have 10 different choices.

```python
validate_epoch(simple_net)
```




    0.1217



Let's bundle up our weights (also called parameters or params) into one variable, so we can easily iterate through all of them when we train the model

```python
params = w1, w2, b1, b2
#for p in params:
   # p.grad.zero_()
```

Let's define how we calculate the gradient. It's quite mathematical but in PyTorch luckily we don't need to do this by hand.
The way the gradient is calculated, is that it goes backward through all the functions were it is called and that is kept track off by PyTorch. 
To summarize it in short, we first do a prediction, with those predictions we calculate the loss. By doing loss.backward(), we calculate the gradients which are then stored inside the parameters variables. The gradients tell us which way we should update the weights in order to get a lower loss next time. (i.e. better predictions)

```python
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = my_mnist_loss(preds, yb)
    loss.backward()
```

This is the function we will use to train the model for one epoch, i.e. going through all pictures in the dataset once. (using batches of 12 that we specified before)
We ask xb and yb from the dataloaders, we calculate the gradients for that batch and then we update the parameters (weights and biases). In order to specify by how much we change the parameters, we set a learning rate (variable lr)

```python
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```

```python
lr = 0.1
```

This is where we first train our model for one epoch. Exciting! :)

```python
train_epoch(simple_net, lr, params)
```

Now let's see if your accuracy has already improved: (I get around 86%, your results should say something similar). Yes it improved quite spectacularly. This is pretty magic!

```python
validate_epoch(simple_net)
    
```




    0.8666



Now let's do the same thing twenty times. (20 epochs)

```python
for i in range(20):
    train_epoch(simple_net, lr, params)
    print(validate_epoch(simple_net), end=' ')
```

    0.9015 0.9092 0.9211 0.9225 0.9127 0.9288 0.9331 0.9368 0.933 0.9376 0.9381 0.941 0.9397 0.9403 0.9413 0.9441 0.9463 0.9488 0.9472 0.9457 

As you can see, the accuracy has improved into the nineties! I\'m running this on a CPU on my personal laptop (6 years old Macbook Pro) and it takes less than a minute or so. I find it pretty amazing to get these kinds of results with a relatively simple neural net that doesn't take lots of computing power. I can't wait to learn more about deep learning and all of its applications!

I hope you enjoyed this article and if you want to contact me or give feedback on this exercise, feel free!
