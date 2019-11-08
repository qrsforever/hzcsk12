import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from subprocess import CalledProcessError
from time import sleep
import time

from visdom import Visdom

from flask import Flask, request
from cauchy.utils.helpers.visdom_helper import find_free_port, free_port
from cauchy.utils.tools.webservices import gen_custom_model, check_viz_status
from cauchy.utils.helpers.file_helper import FileHelper


def msg(content, code):
  resultDic = {}
  resultDic['content'] = content
  resultDic['code'] = code
  repJsonStr = json.dumps(resultDic)
  return repJsonStr


app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/cauchy/train', methods=['POST', 'GET'])
def train():
  # parse params
  try:
    reqJsonDic = json.loads(request.get_data().decode())
  except Exception as e:
    return msg(str(e), '100001')

  try:
    net_def = reqJsonDic['net_def']
    viz_pid = reqJsonDic['viz_pid']
    viz_port = reqJsonDic['viz_port']
    project_dir = reqJsonDic['project_dir']

    if net_def != "":
      print(net_def)
      gen_custom_model(net_def, project_dir)
    hypes = reqJsonDic['hypes']

    # set up tmp dir for hypes.json
    tmp_dir = FileHelper.tmp_dir()
    while os.path.exists(tmp_dir):
      tmp_dir = FileHelper.tmp_dir()
    FileHelper.make_dirs(tmp_dir)

    hypes_path = os.path.join(tmp_dir, "hypes.json")
    with open(hypes_path, 'w') as fout:
      fout.write(hypes)

    # print(reqJsonDic)

    sub_task = subprocess.Popen([
        "python", "run_tasks.py", "--hypes", hypes_path, "--viz_pid",
        str(viz_pid), "--viz_port",
        str(viz_port), "--tmp_dir", tmp_dir
    ])
    return msg({"task_id": str(sub_task.pid)}, "100200")

  except Exception as e:
    # kill visdom
    subprocess.run(["kill", "-9", viz_pid])
    print(e)
    return msg(str(e), "100002")

@app.route('/cauchy/free_port', methods=['POST', 'GET'])
def get_port():
  viz_port = str(find_free_port(8140, 8199))
  try:
    viz_instance = subprocess.Popen(
        ["python", "-m", "visdom.server", "-port", viz_port])
    sleep(2)
    return msg({'viz_pid': viz_instance.pid, "viz_port": viz_port}, "100200")
  except Exception as e:
    print(str(e))
    return msg(str(e), "100003")


@app.route('/cauchy/stop_training', methods=['POST', 'GET'])
def stop_training():
  try:
    reqJsonDic = json.loads(request.get_data().decode())
  except Exception as e:
    return msg(str(e), '100001')

  try:
    task_pid = reqJsonDic['task_pid']
    viz_pid = reqJsonDic['viz_pid']
    subprocess.run(["kill", "-9", task_pid])
    subprocess.run(["kill", "-9", viz_pid])
    return msg("Training Stopped", "100200")
  except Exception as e:
    print(e)
    return msg(str(e), "100004")


@app.route('/cauchy/stop_visdom', methods=['POST', 'GET'])
def stop_visdom():
  try:
    reqJsonDic = json.loads(request.get_data().decode())
  except Exception as e:
    return msg(str(e), '100001')

  try:
    viz_pid = reqJsonDic['viz_pid']
    subprocess.run(["kill", "-9", viz_pid])
    return msg("Visdom Stopped", "100200")
  except Exception as e:
    print(e)
    return msg(str(e), "100005")


@app.route('/cauchy/test_single', methods=['POST', 'GET'])
def test_single():
  # parse params
  try:
    reqJsonDic = json.loads(request.get_data().decode())
  except Exception as e:
    return msg(str(e), '100001')

  try:
    keys = ["hypes", "phase", "resume", "img_content", "viz_pid", "viz_port"]
    vals = [reqJsonDic[_key] for _key in keys]
    # set up tmp dir for hypes.json
    tmp_dir = FileHelper.tmp_dir()
    while os.path.exists(tmp_dir):
      tmp_dir = FileHelper.tmp_dir()
    FileHelper.make_dirs(tmp_dir)

    # write hypes
    hypes_path = os.path.join(tmp_dir, "hypes.json")
    with open(hypes_path, 'w') as fout:
      fout.write(vals[0])

    # write img
    test_img_path = os.path.join(tmp_dir, "test.png")
    with open(test_img_path, 'wb') as fout:
      fout.write(base64.b64decode(vals[3].split(",")[1]))

    sub_task = subprocess.Popen([
        "python", "run_tasks.py", "--hypes", hypes_path, "--phase", vals[1],
        "--resume", vals[2], "--test_img", test_img_path, "--viz_pid",
        str(vals[4]), "--viz_port",
        str(vals[5]), "--tmp_dir", tmp_dir
    ])
    return msg({"task_id": str(sub_task.pid)}, "100200")
  except Exception as e:
    print(e)
    subprocess.run(["kill", "-9", vals[-2]])
    return msg(str(e), "100006")


@app.route('/cauchy/test_batch', methods=['POST', 'GET'])
def test_batch():
  # parse params
  try:
    reqJsonDic = json.loads(request.get_data().decode())
  except Exception as e:
    print(e)
    return msg(str(e), '100001')

  try:
    keys = ["hypes", "phase", "resume", "test_dir", "viz_pid", "viz_port"]
    vals = [reqJsonDic[_key] for _key in keys]
    # set up tmp dir for hypes.json
    tmp_dir = FileHelper.tmp_dir()
    while os.path.exists(tmp_dir):
      tmp_dir = FileHelper.tmp_dir()
    FileHelper.make_dirs(tmp_dir)

    hypes_path = os.path.join(tmp_dir, "hypes.json")
    with open(hypes_path, 'w') as fout:
      fout.write(vals[0])
    sub_task = subprocess.Popen([
        "python", "run_tasks.py", "--hypes", hypes_path, "--phase", vals[1],
        "--resume", vals[2], "--test_dir", vals[3], "--viz_pid",
        str(vals[4]), "--viz_port",
        str(vals[5]), "--tmp_dir", tmp_dir
    ])
    return msg({"task_id": str(sub_task.pid)}, "100200")
  except Exception as e:
    print(e)
    subprocess.run(["kill", "-9", vals[-2]])
    return msg(str(e), "100006")


@app.route('/cauchy/check_viz_started', methods=['POST', 'GET'])
def check_viz_started():
  # parse params
  try:
    reqJsonDic = json.loads(request.get_data().decode())
  except Exception as e:
    print(e)
    return msg(str(e), '100001')

  viz_port = int(reqJsonDic['viz_port'])
  viz_host = int(reqJsonDic['viz_host'])
  return check_viz_status(viz_host, viz_port)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--port',
      default=None,
      type=int,
      dest='port',
      help="port to run cauchy services")

  args = parser.parse_args()

  app.run(host="0.0.0.0", port=args.port)
