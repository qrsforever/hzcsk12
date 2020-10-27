#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import base64
import smtplib
from email.mime.text import MIMEText
from urllib.request import urlopen

DEBUG = False

def debug_out(fname, data):
    if DEBUG:
        with open('/tmp/%s.txt' % fname, 'a') as fw:
            fw.write(data)
            fw.write('\n')

def main():
    sender = os.environ.get('EMAIL_SENDER', None)
    passwd = os.environ.get('EMAIL_PASSWD', None)
    recver = os.environ.get('EMAIL_RECVER', None)
    if sender is None or passwd is None or recver is None:
        return

    host = os.environ.get('CONSUL_ADDR', None)
    port = os.environ.get('CONSUL_PORT', None)
    if host is None or port is None:
        return

    response = urlopen('http://{}:{}/v1/status/leader'.format(host, port))
    if host not in response.read().decode('utf-8'):
        return

    data = json.load(sys.stdin)
    if len(data) == 0:
        return

    debug_out('errors', json.dumps(data))

    for item in data:
        key = item['Key']
        value = base64.b64decode(item['Value'].encode()).decode()

        msg = MIMEText(value, 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = recver
        msg['Subject'] = '[TalentAI异常] %s' % key

        debug_out('errors_key', key)
        debug_out('errors_value', value)

        try:
            smtpObj = smtplib.SMTP('smtp.qq.com')
            smtpObj.login(sender, passwd)
            smtpObj.sendmail(sender, recver.split(','), msg.as_string())
            smtpObj.quit()
            os.system('curl --request DELETE http://{}:{}/v1/kv/{}'.format(host, port, key))
        except smtplib.SMTPException as e:
            debug_out('errors_except', '%s'.format(e))
        except Exception as e:
            debug_out('errors_except', '%s'.format(e))

if __name__ == "__main__":
    main()
    print('0')
