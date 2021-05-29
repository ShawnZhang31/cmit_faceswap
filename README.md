# Face Swap

面部融合应用, 效果如下:     

<table>
  <tr>
     <td align="center">原图</td>
     <td align="center">合成模板</td>
     <td align="center">合成效果图</td>
  </tr>
  <tr>
    <td><img src="./docs/liushishi.png" width=100%></td>
    <td><img src="./docs/female.jpg" width=100%></td>
    <td><img src="./docs/merge.jpg" width=100%></td>
  </tr>
 </table>

<!-- <p float="left">
  <img src="./docs/liushishi.png" width="255" />
  <img src="./docs/female.jpg" width="200" /> 
  <img src="./docs/merge.jpg" width="200" />
</p> -->

<!-- <div style="text-align: center; ">
    <ul style="display: table">
        <li style="display: table-cell; list-style: none; vertical-align: middle; width:34%"><img src="./docs/liushishi.png" alt="source"></li>
        <li style="display: table-cell; list-style: none; vertical-align: middle; font-size: xx-large; ">+</li>
        <li style="display: table-cell; list-style: none; vertical-align: middle;width:33%"><img  src="./docs/female.jpg" alt="mask"></li>
        <li style="display: table-cell; list-style: none;  vertical-align: middle; font-size: xx-large">=</li>
        <li style="display: table-cell; list-style: none; vertical-align: middle;width:33%"><img  src="./docs/merge.jpg" alt="result"></li>
    </ul>
    <h6>人脸融合示意</h6>
</div> -->

## 1. 部署说明
将代码克隆到部署服务器上，按照如下步骤配置自己的部署工程:   
- Docker部署
- supervisor+gunicorn部署
### 1.1 Docker部署

docker部署默认打印的gunicorn日志为debug日志，生产环境下请修改`boot.sh`中的日志级别    

- 生产环境的gunicorn移除`--log-level=debug --preload`   

```bash
#!/bin/sh

# 日志文件
touch /cmit_faceswap/log/access.log
touch /cmit_faceswap/log/error.log
# touch /cmit_faceswap/log/output.log

exec gunicorn -b 0.0.0.0:5000 -w 2 cmit_faceswap:app --access-logfile=/cmit_faceswap/log/access.log --error-logfile=/cmit_faceswap/log/error.log --timeout=180 --log-level=debug --preload
```
#### 1.1.1 部署要求
- Docker 18.03+
#### 1.1.2 配置面部融合的模板
将面部融合的模板放到`./res/templates`目录下面，并更新`./res/templates/templates.yaml`文件中的模板配置

如：该项目中配置了演示用模板template1，并在模板将template1的配置信息写入`./res/templates/templates.yaml`文件中      
```yaml
template1: # 模板名称，这个名称非常重要，后面调用接口的时候需要使用这个名称来查找对应的模板文件
  male: # 男性的模板配置
    img: './res/templates/template1/male/male.jpg'  # 带有头发的模板
    face: './res/templates/template1/male/male_no_hair.jpg' # 没有带头发的模板
    hair: './res/templates/template1/male/hair.jpg' # 头发模板，注意头发意外的部分全部应设置为白色
  female: # 女性的模板配置
    img: './res/templates/template1/female/female.jpg'  # 带有头发的模板
    face: './res/templates/template1/female/female_no_hair.jpg' # 没有带头发的模板
    hair: './res/templates/template1/female/hair.jpg' # 头发模板，注意头发意外的部分全部应设置为白色
```
#### 1.1.3 配置环境变量
在`./.env`文件中设置环境变量，下面是项目中自带一个.env文件，如果懒得的话，可以只替换一下`SECRET_KEY`        

```env
APP_CONFIG=production #部署环境：production-生产环境;development-开发环境；testing-测出环境
SECRET_KEY=fadfascsvasdfahudquerw22wxvZf    # 墙裂建议生产部署的适合替换SECRET_KEY

DLIB_FACE_LANDMARK_SHAPE_FILE_PATH=./res/dlib/shape_predictor_68_face_landmarks.dat #dlib face landmark模型路径
GENDER_PROTOTXT_FILE_PATH=./res/gender/gender_deploy.prototxt   # Caffe性别识别模型路径
GENDER_NET_FILE_PATH=./res/gender/gender_net.caffemodel # Caffe性别识别模型路径

FLASK_APP=cmit_faceswap.py  # flask应用入口入口
FLASK_ENV=production    # flask应用模式
TEMPLATES_ROOT=./res/templates  # 模板文件的根目录
TEMPLATES_CONFIG_NAME=templates.yaml    # 模板配置信息
```

#### 1.1.4 启动docker容器
`./docker-compose.yml`文件是docker的默认配置文件，具体的配置如下:   
```yaml
version: '3.7'
services: 
    webapp:
        build: .
        ports: 
            - "5000:5000"   # 宿主机端口:docker flask应用端口;可以通过修改该配置将docker flask应用的5000端口，映射到宿主机上的其他端口
        env_file: 
            - .env
        restart: always
        volumes: 
            - ./logs/:/cmit_faceswap/log    # 同步docker flask应用的日志到宿主机的./logs目录
            - ./res/:/cmit_faceswap/res     # 同步docker flask应用的res目录到宿主机的./res目录，容器启动之后可以在宿主机的./res目录下更新flask应用的res资源文件
```

一切配置完成之后，在项目根目录执行`docker-compose up -d`即可启动容器；首次启动需要自动安装依赖文件，可能耗时较长      
访问http://{ip}:5000，看到如下页面，表示容器启动成功:       
![index](./docs/index.png)

### 1.2 supervisor+gunicorn部署
以CentOS 7.8 为例

#### 1.2.1 配置部署环境

- 安装前置依赖
```bash
sudo yum -y install epel-release
sudo yum -y install git gcc gcc-c++ cmake3
sudo yum install -y python34 python34-devel python34-pip
sudo yum install -y python python-devel python-pip
sudo yum -y install python-devel numpy python34-numpy
sudo yum -y install gtk2-devel
sudo yum install -y libpng-devel
sudo yum install -y jasper-devel
sudo yum install -y openexr-devel
sudo yum install -y libwebp-devel
sudo yum -y install libjpeg-turbo-devel
sudo yum install -y freeglut-devel mesa-libGL mesa-libGL-devel
sudo yum -y install libtiff-devel
sudo yum -y install libdc1394-devel
sudo yum -y install tbb-devel eigen3-devel
sudo yum -y install boost boost-thread boost-devel
sudo yum -y install libv4l-devel
sudo yum -y install gstreamer-plugins-base-devel
```

- 升级pip3到最新版本
```bash
$ sudo python3 -m pip install --upgrade pip
```

- 安装virtualenv
```bash
$ sudo python3 -m pip install virtualenv
```

#### 1.2.2 创建并激活项目虚拟环境
- 在项目根目录创建项目虚拟环境    
```bash
$ python3 -m virtualenv venv
```

- 激活项目虚拟环境
```bash
$ source venv/bin/activate
```

接下来的操作根据bash并且前的符号来确定是在虚拟环境中执行，还是在普通环境中执行：    
- `$`表示在普通环境中执行
- `(venv) [xxxx]$`表示在虚拟环境中执行
- `deactivate`: 退出虚拟环境
- `source venv/bin/activate`: 激活虚拟环境

#### 1.2.3 在虚拟环境中安装项目依赖文件

安装项目依赖的包
```bash
(venv)[xxxx]$ pip install -r requirements.txt
```

**PS：由于CentOS在图像开发上的支持度很差，此时可能会出现dlib setup.py的错误，没有什么好的解决办法，并根据具体的错误Google了**

#### 1.2.4 安装并配置supervisor
`deactivate`退出虚拟环境    
- 安装supervsior    
```bash
$ sudo yum install supervisor -y
```

- 设置supervisor为开机启动
```bash
$ sudo systemctl enable supervisord
```

- 查看supervisor的状态
```bash
$ sudo systemctl status supervisord
```

- 启动supervisor
```bash
$ sudo systemctl start supervisord
```





- 配置supervisor


```
[program:cmit_faceswap]
command=/{项目根目录}/venv/bin/gunicorn -b 0.0.0.0:5000 -w 2 cmit_faceswap:app --timeout=180 --log-level=debug --preload; supervisor启动命令
directory=/{项目根目录}                                                ; 项目的文件夹路径
startsecs=0                                                                             ; 启动时间
stopwaitsecs=0                                                                          ; 终止等待时间
autostart=true                                                                         ; 是否自动启动
autorestart=true                                                                       ; 是否自动重启
stdout_logfile=/{项目根目录}/logs/gunicorn.log                           ; log 日志
stderr_logfile=/{项目根目录}/logs/gunicorn.err  
```

- 启动supervisor

```bash
$ supervisord -c supervisor.conf 
```

- supervisor的其命令
  - 察看supervisor的状态
    ```bash
    supervisorctl -c supervisor.conf status                    察看supervisor的状态
    ```
  - 重新载入配置文件
    ```bash
    supervisorctl -c supervisor.conf reload                    重新载入配置文件
    ```


## 2. 接口说明
### 2.1 /api/v1/faceswap
v1版本的面部融合接口
- 请求方式**POST**  

- 参数说明

参数 | 类型 | 是否必填 | 参数说明
:----- | :----- | :----- | :-----
image_ref | base64 str | 是 | 用于合成的图像，使用Base64对图像进行编码；目前仅支持通道为1、3、4的图像；最好提交jpg/jpeg格式的图像，人脸最好正对摄像头
template_name | str | 是 | 提前配置进应用中的模板的名称

- 返回结果      

参数 | 类型 | 是否必有 | 参数说明
:----- | :----- | :----- | :-----
code | int | 是 | api请求结果状态码：200-请求成功；4100-请求参数有误；5300-服务端错误；
error | str/null | 是 | 请求失败的具体错误信息；当请求成功时为null
swaped_image | base64 str| 是| 面部融合的结果，为Base64编码的jpg图像；当请求失败的时候为null

## 3. TODO 
- [ ] 为v1版本的面部融合方案添加颜色高斯融合方式，解决面部高光区域对Seamless融合算法的影响；
- [ ] 增加v2融合方案，使用面部468点识别方案扩大面部融合区域、优化颜色融合方案


