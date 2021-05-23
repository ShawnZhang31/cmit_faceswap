"""
检验用于提交的图片数据是否符合遥要求
"""
import json
from functools import wraps
from flask import request, current_app

from webapp.utils.API_RESPONE_CODE import API_RESPONE_CODE


def request_required_params(params_list):
    """检查用于提交的图像是否合乎要求"""
    def request_required_params_decorator(f):
        @wraps(f)
        def decorate_function(*args, **kwargs):
            resp={}
            for key in params_list:
                if key not in request.form:
                    resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
                    resp['error'] = key+"是必填参数"
                    resp['swaped_image']=None
                    return json.dumps(resp)
                else:
                    kwargs[key]=request.form.get(key)
            return f(*args, **kwargs)
        return decorate_function
    return request_required_params_decorator
