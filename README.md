# Triton Inference Server - IXRT backend

使用TensorRT 8.4.1.5的接口进行适配  
可以处理动态形状的推理请求

## 基础概念
关于triton的基础概念，请参考NVIDIA提供的[triton tutorials](https://github.com/triton-inference-server/tutorials)

## 前置要求
首先确保，您的工作环境中已经正确安装了tritonserver  
例如，可以在/opt/tritionserver/bin/目录下，找到可执行文件tritonserver  
triton inference server 依赖库的组织如下  
```
/opt/tritonserver/bin
                  /lib
                  /backend
```
如需安装triton inference server，请参考[triton 相关文档](http://10.150.9.95/docs/4-sdk-docs-mr-x86-zh/_source/tis.html#)  
获取相关docker或者从源码编译
## 配置环境
### 安装IXRT 运行时（必选）
可以从如下网址获取run安装包  
http://10.150.9.95/swapp/release/ixrt/latest/  
这个地址下放置的安装文件是每日更新的，请注意版本号下文中的版本号更新，截止本文更新时，最新的IXRT版本号是0.7.0  
我们以ixrt-0.7.0+corex.latest.version-linux_x86_64.run版本为例，可以进入到任意路径，例如/tmp  
下载run文件，如下  
```
wget http://10.150.9.95/swapp/release/ixrt/latest/ixrt-0.7.0+corex.latest.version-linux_x86_64.run
```
添加使用权限
```
chmod +x ixrt-0.7.0+corex.latest.version-linux_x86_64.run 
```
运行安装过程
```
./ixrt-0.7.0+corex.latest.version-linux_x86_64.run 
```
安装结束后，输出信息提示，头文件和库文件的安装位置，常见如下 
```
Welcome to use IxRT! Now installing IxRT library to /usr/local/corex...
Thank you for using IxRT!
- library path: /usr/local/corex/lib
- header path: /usr/local/corex/include
- samples path: /usr/local/corex/samples/ixrt
```

### 安装IXRT Python包（可选）
为了方便脚本化测试，和Python接口的使用，请根据您所使用机器的Python版本选择安装包  
例如，Python版本为3.7，可以下载安装包
```
wget http://10.150.9.95/swapp/release/ixrt/latest/ixrt-0.7.0+corex.latest.version-cp37-cp37m-linux_x86_64.whl
```
安装Python包
```
pip install ixrt-0.7.0+corex.latest.version-cp37-cp37m-linux_x86_64.whl
```
关于IXRT的使用细节，请参考[IXRT使用文档](http://10.150.9.95/swapp/docs/ixrt/dev/index.html)
### 编译IXRT triton backend
#### 下载源码库
请进入您的工作目录，例如，在/home/xxx/
```
cd /opt
```
获取工程源码
```
git clone ssh://git@bitbucket.iluvatar.ai:7999/swapp/tensorrt_backend.git
```
现在，请进入源码目录
```
cd /home/xxx/tis_ixrt_backend
```

#### 配置编译选项
打开工程目录中的build.sh  
修改-DIXRT_HOME=/path_to_ixrt  
指向IXRT的头文件，以及库安装路径  
以上面的IXRT安装过程为例，可以修改为
```
-DIXRT_HOME=/usr/local/corex/
```

#### 执行编译过程
添加执行权限
```
chmod +x build.sh
```

执行编译脚本
```
./build.sh
```
编译过程会自动在当前工程目录下，创建build文件夹，所有的编译产物放置在这里

### 配置模型仓库
模型仓库是triton服务启动时需要加载的模型文件  
组织形式是
```
<model_repository>/<model_name>/<version_directory>/ixrt_engine_file
                               /config.pbtxt
```
如果有多个类型的模型，表达形式为
```
<model_repository>/<model_name_1>/<version_directory>/ixrt_engine_file
                                 /config.pbtxt
                  /<model_name_2>/<version_directory>/ixrt_engine_file
                                 /config.pbtxt
```
IXRT后端可接受的模型文件是根据模型生成的engine文件  
#### 使用已有模型仓库
为了方便您能够快速体验IXRT的后端，已经准备好ResNet-18的FP16精度，分类网络测试用例，允许输入的形状尺寸是[1,3,112,112]~[16,3,448,448]
下载文件包
```
wget http://10.150.9.95/swapp/projects/ixrt/data/tis_ixrt_backend_data.zip
```
解压缩文件包
```
unzip tis_ixrt_backend_data.zip
```
其中包含的文件夹有
* ixrt_model_repository：测试模型和配置文件
* resnet18：测试用图片文件
例如，将将ixrt_model_repository文件夹移动到/opt目录
按照如上路径配置要求，您将会看到这样的文件结构
```
/opt/ixrt_model_repository/cls/1/resnet18_dynamic.engine
                              /config.pbtxt
```
config文件的内容，请参考[说明文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
#### 自行构建模型仓库
关于IXRT engine的详细信息请参考[文档说明](http://10.150.9.95/swapp/docs/ixrt/dev/python_api.html#engine)
在scripts文件夹下，提供了一个onnx文件转为FP16精度engine的脚本示例-onnx2engine.py  
由于需要使用到ixrt的Python接口，请安装IXRT的Python包  
生成engine的功能，很快会将这个功能集成在[ixrtexec工具](http://10.150.9.95/swapp/docs/ixrt/dev/ixrtexec.html)
下面展示从ResNet-18的ONNX模型，构建动态形状engine的流程  
获取ResNet-18的动态形状文件，下载数据
```
wget http://10.150.9.95/swapp/projects/ixrt/data/resnet18.zip
```
解压缩
```
unzip resnet18.zip
```
例如，文件解压到/home/data/resnet18  

在resnet18文件夹下，resnet18-all-dynamic.onnx支持batch size，image height， image width的动态变化  

修改onnx2engine.py文件的4~6行，指定ONNX所在位置（/home/data/resnet18)，输入ONNX文件名(resnet18-all-dynamic.onnx)和输出engine文件名(esnet18_dynamic.engine)  
```
dir_path="/home/data/resnet18/"
onnx_name = 'resnet18-all-dynamic.onnx'
engine_name = "resnet18_dynamic.engine"
```
设置动态范围，第17行
```
profile.set_shape("input", Dims([1,3,112,112]), Dims([2,3,224,224]), Dims([16,3,448,448]))
```
如上语句，针对动态输入接口"input"，设置最小尺寸[1,3,112,112]，最常用尺寸[2,3,224,224]，最大尺寸[16,3,448,448]  
执行Python文件
```
python onnx2engine.py
```
engine文件会出现在/home/data/resnet18/resnet18_dynamic.engine
### 安装后端库
成功编译之后，在当前工程目录的build文件夹下，可以找到名字为libtriton_ixrt.so的文件  
可以放置于如下三个路径，会被triton 服务端程序找到  
* <model_repository>/<model_name>/<version_directory>/libtriton_ixrt.so

* <model_repository>/<model_name>/libtriton_ixrt.so

* <global_backend_directory>/backend_type/libtriton_ixrt.so

如果是全局路径，一般放置在triton的安装目录下，如下所示
```
/opt/
  tritonserver/
    backends/
      ixrt/
        libtriton_ixrt.so
        ... # other files needed by mybackend
```

### 安装Python Client
使用Python客户端，可以方便的发送推理请求，并进行测试  
使用如下安装命令
```
pip install tritonclient[http] opencv-python-headless
```

## 运行测试
测试代码位于python_client_example  
tis_ixrt_backend_data.zip中的resnet18文件夹下有三张图片  
可以将resnet18文件夹放置到期望位置
修改python_client_example/client.py的127行，指向resnet18文件夹位置  
测试用例展示的内容有：
* 输入1x3x224x224图片，展示推理流程
* 输入1x3x196x196图片，展示image height， image width动态调整后的推理
* 输入2x3x224x224图片，展示batch size动态调整后的推理
使用如下命令启动triton 服务
```
tritonserver --model-repository=/your_path/ixrt_model_repository/
```
如果使用前文所示例的模型仓库位置，命令如下
```
tritonserver --model-repository=/opt/ixrt_model_repository/
```
然后运行客户端，查看demo演示
```
python python_client_example/client.py
```

期望的输出结果为
```
------------------------------Python inference result------------------------------
Top 1:   10.1484375  n02123159 tiger cat
Top 2:   9.9296875  n02123045 tabby, tabby cat
Top 3:   9.6640625  n02124075 Egyptian cat
Top 4:   8.6171875  n03887697 paper towel
Top 5:   8.2578125  n03958227 plastic bag
------------------------------Python inference result------------------------------
Top 1:   11.171875  n02123159 tiger cat
Top 2:   10.765625  n02123045 tabby, tabby cat
Top 3:   10.3125  n02124075 Egyptian cat
Top 4:   10.2578125  n02127052 lynx, catamount
Top 5:   8.1171875  n02123394 Persian cat


batch result
result  0
------------------------------Python inference result------------------------------
Top 1:   10.1484375  n02123159 tiger cat
Top 2:   9.9296875  n02123045 tabby, tabby cat
Top 3:   9.6640625  n02124075 Egyptian cat
Top 4:   8.6171875  n03887697 paper towel
Top 5:   8.2578125  n03958227 plastic bag
result  1
------------------------------Python inference result------------------------------
Top 1:   16.765625  n01558993 robin, American robin, Turdus migratorius
Top 2:   12.3046875  n01824575 coucal
Top 3:   11.3984375  n02486261 patas, hussar monkey, Erythrocebus patas
Top 4:   9.125  n01560419 bulbul
Top 5:   9.125  n01807496 partridge
```

