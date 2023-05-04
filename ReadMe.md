## 项目说明
这是你只看一次（YOLO）版本3的中文重写。
### 训练时间
我电脑1080ti+i9 9900k，一个老气的配置。8.2万张图片训练一个轮回要2个小时左右，训练100轮回需要200多个小时，不过4-5个轮回就能看到一些效果。
### 训练用数据
我训练时用的数据来自[cocodataset](https://cocodataset.org/#download)，版本是2014版的数据集，该网址有多个图片数据集，我建议复制下载链接后使用下载器下载。
### 一些简要说明
* 这一次文件夹命名用不同名词表示一个意思，虽然我不建议大家这么做，但我感觉挺有趣的。
* 命名可能存在错误，这是由于有些公式我没能理解意味着什么。
* [数据屋/样例](%E6%95%B0%E6%8D%AE%E5%B1%8B%2F%E6%A0%B7%E4%BE%8B)文件夹直接放入图片就行。
* [训练.py](%E8%AE%AD%E7%BB%83.py)生成的检查点会在[检查点居室](%E6%A3%80%E6%9F%A5%E7%82%B9%E5%B1%85%E5%AE%A4)里；而[测试.py](%E6%B5%8B%E8%AF%95.py)的结果会在[输出间/样例](%E8%BE%93%E5%87%BA%E9%97%B4%2F%E6%A0%B7%E4%BE%8B)里，会在图片中直接显示识别的结果，前提是你已经在[数据屋/样例](%E6%95%B0%E6%8D%AE%E5%B1%8B%2F%E6%A0%B7%E4%BE%8B)中放入了图片。
* [图片箱子](%E6%95%B0%E6%8D%AE%E5%B1%8B%2Fcoco%2F%E5%9B%BE%E7%89%87%E7%AE%B1%E5%AD%90)和[标签箱子](%E6%95%B0%E6%8D%AE%E5%B1%8B%2Fcoco%2F%E6%A0%87%E7%AD%BE%E7%AE%B1%E5%AD%90)里放`训练集2014`和`验证集2014`文件夹，存放相应的图片和标签。
* 一个检查点236兆，我提供了一个[检查点](https://huggingface.co/zozero/YOLO_chinese/tree/main)，这个检查点效果很糟糕，因为训练次数太少了。
### 使用
把数据放在相应的位置，直接运行即可。
### 问题
1. 损失值太大时（大概200多）`非极大值抑制`函数的while循环会无限循环，因为计算的张量中有一些错误的值；只要损失值不大就不会有这个问题。
2. 变量`记录者`我注释掉了，因为会有中文路径报错的问题和会导致训练速度变慢。解决方法是使用一个没有中文的路径；训练速度问题我建议使用异步操作解决，当然我没有去解决这个问。
3. 损失值大的话也会导致输出的[样例](%E8%BE%93%E5%87%BA%E9%97%B4%2F%E6%A0%B7%E4%BE%8B)图片出现空白填充，损失值衰减后就会大幅度减少这个问题。