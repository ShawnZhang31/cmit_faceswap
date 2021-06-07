"""
检查模板
"""
import os
import json
from functools import wraps
from flask import request, current_app
import yaml

from webapp.utils.API_RESPONE_CODE import API_RESPONE_CODE

def getTemplateConfig():
    """
        获取人脸合成的模板配置
    """
    TEMPLATES_ROOT = os.getenv("TEMPLATES_ROOT") or './res/templates'
    TEMPLATES_CONFIG_NAME = os.getenv('TEMPLATES_CONFIG_NAME') or 'templates.yaml'
    TEMPLATES_CONFIG_FILE_PATH = os.path.join(TEMPLATES_ROOT, TEMPLATES_CONFIG_NAME)
    templates = None
    with open(TEMPLATES_CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        templates = yaml.load(f, Loader=yaml.SafeLoader)
        f.close()
    return templates

def getTemplateImgs(template_name):
    """获取用于提交的模板数据"""
    templates = getTemplateConfig()
    if not templates.__contains__(template_name):
        return None
    else:
        return templates[template_name]

def tempate_required_withkey(tempalte_key):
    """获取提交的模板的配置数据"""
    def tempate_required_withkey_decorator(f):
        @wraps(f)
        def decorate_function(*args, **kwargs):
            resp={}
            tempalte_name = request.form.get(tempalte_key)
            templates = getTemplateImgs(tempalte_name)
            if templates is None:
                resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
                resp['error'] = "未找到"+tempalte_name+"的相关配置文件"
                resp['swaped_image']=None
                return json.dumps(resp)
            else:
                kwargs['template']=templates
                kwargs['template_name']=tempalte_name
            return f(*args, **kwargs)
        return decorate_function
    return tempate_required_withkey_decorator
