# -*- coding:gbk -*-
'''
Created on 2017.10.24.

@author: l00198668
'''

import os
import sys
import re
import time
import platform
import paramiko
from scp import SCPClient


class SSHConnection(object):
    def __init__(self, host, port, username, password):
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._transport = None
        self._sftp = None
        self._client = None
        self._scp = None
        self._connect()

    def _connect(self):
        transport = paramiko.Transport((self._host, self._port))
        transport.connect(username=self._username, password=self._password)
        self._transport = transport

    # sftp下载
    def get(self, remotepath, localpath):
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.get(remotepath, localpath)

    # sftp上传
    def put(self, localpath, remotepath):
        if self._sftp is None:
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        self._sftp.put(localpath, remotepath)

    # scp下载
    def scp_get(self, remotepath, localpath):
        if self._scp is None:
            self._scp = SCPClient(self._transport)
        self._scp.get(remotepath, localpath)

    # scp上传
    def scp_put(self, localpath, remotepath):
        if self._scp is None:
            self._scp = SCPClient(self._transport)
        self._scp.put(localpath, remotepath)

    # 执行命令
    def exec_command(self, command):
        if self._client is None:
            self._client = paramiko.SSHClient()
            self._client._transport = self._transport
        stdin, stdout, stderr = self._client.exec_command(command)
        data = stdout.read()
        if len(data) > 0:
            print(data.strip())
            return data
        err = stderr.read()
        if len(err) > 0:
            print(err.strip())
            return err

    def close(self):
        if self._transport:
            self._transport.close()
        if self._client:
            self._client.close()

class SSH():

    def __init__(self):
        self.handle = None
        self.encoding = 'utf-8'  # 'ascii', 'utf-8', 'latin-1'

    def enc(self, msg):
        return bytes(msg, encoding=self.encoding)

    def dec(self, byte):
        try:
            msg = str(byte, encoding=self.encoding)
            msg = re.sub("\x1b\[[\d+;]*m[\x0f]*", "", msg)
            if re.search(r"[\x80-\xff]", msg):
                msg = "\n"
        except:
            msg = "\n"
        return msg

    def ping_server(self, ip):
        msg = platform.system()
        if msg.find('Linux') >= 0:
            ret = os.system("ping %s -c 1 -W 1" % ip)
        elif msg.find('Windows') >= 0:
            ret = os.system("ping %s -n 1 -w 1000" % ip)
        else:
            ret = -1
        print('[ping_server] %s = %d' % (ip, ret))

        return ret

    def init(self, ip, user, passwd):
        msg = ''

        self.handle = paramiko.SSHClient()
        self.handle.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #if self.ping_server(ip) == 0:
        self.handle.connect(ip, username=user, password=passwd)
        self.channel = self.handle.invoke_shell()
        msg = self.recv(':~', 30)
            #print('wxzwxzwxzwxz')
            #print(msg)
            #else:
        '''
            self.handle.connect('10.71.200.40', username='l00198668', password='l00198668')   
            self.channel = self.handle.invoke_shell()     
            msg = self.test('ssh root@192.168.2.114', 'password')
            if msg.find('yes/no') > 0:
                msg = self.test('yes', 'password')
            if msg.find('password') > 0:
                self.test('root', ':~')
            self.recv(':~')
        '''
            #print('error: ssh server offline')
            #'''

        return msg

    def close(self):
        try:
            self.channel.send('exit\n')
            time.sleep(1)
            self.handle.close()
        except:
            print('error: ssh close')

    def send(self, cmd):
        msg = ''
        try:
            self.channel.send(cmd)
        except:
            print('error: ssh send')

        return msg

    def recv(self, expect='', tout=10):
        msg = ''
        idx = -1
        err = 0
        i = 0

        try:
            tout = int(tout*100) if tout>0 else 1
            for i in range(tout*10):
                time.sleep(0.01)
                if(self.channel.recv_ready()):
                    tmp = self.channel.recv(1024)
                    tmp = self.dec(tmp)
                    if(len(tmp) > 0):
                        msg += tmp

                        idx = msg.find(expect)
                        if idx >= 0:
                            break

                    err = 0
                else:
                    err = err + 1
                    if err > tout*1:
                        break
        except:
            print('error: ssh recv')

        '''       
        if len(expect) > 0:
            name = sys._getframe(1).f_code.co_name
            tmp = "[%s] %s = %d(%d)" % (name, expect, idx, i/100)
            print(tmp)
        #'''

        return msg

    def test(self, cmd, expect='', tout=10):
        msg = ''
        flg = 0
        err = 0
        i = 0

        try:
            self.recv('', 0)
            # send and recv
            self.channel.send(cmd + '\n')
            for i in range(tout*1000):
                time.sleep(0.01)
                if(self.channel.recv_ready()):
                    tmp = self.channel.recv(1024)
                    tmp = self.dec(tmp)
                    if(len(tmp) > 0):
                        msg += tmp

                    idx = -1
                    # return contain the cmd
                    if flg == 0:
                        idx = msg.find(cmd[:16])
                        #print(flg, idx, len(cmd))
                        if idx >= 0:
                            idx += len(cmd)
                            flg = 1
                    # the expect should after the cmd string
                    if flg <= 1:
                        idx = msg.find(expect, max(idx, 0))
                        #print(flg, idx)
                        if idx >= 0:
                            flg = 2
                            err = 0
                            break
                    err = 0
                else:
                    err += 1
                    # recv nothing, exit
                    if err > tout*100:
                        print('timeout: ssh test')
                        break
        except Exception as e:
            msg += str(e)
            print(e)

        name = sys._getframe(1).f_code.co_name
        res = ['halt', 'error', 'ok']
        msg += "[%s] %s = %s(%d)" % (name, cmd.strip(), res[flg], i/100)

        return msg


    def su_root(self):
        msg = self.test('su', 'Password')
        if msg.find('yes/no') >= 0:
            msg += self.test('yes', 'Password')
        if msg.find('Password') >= 0:
            msg += self.test('root', '#', 30)
            msg + self.test('cd', ':~')

        return msg

def device(host, cmd, device_ip, timeout=600):

    sh = SSH()
    sh.init(host, 'root', 'root')
    print("ssh host success")
    msg = sh.test('ssh-keygen -f "/root/.ssh/known_hosts" -R %s' % device_ip, ':~')
    print("ssh-keygen device success,cmd is %s." % cmd)
    msg = sh.test('ssh root@%s %s' % (device_ip, cmd), 'Password', 30)
    if msg.find('yes/no') >= 0:
        msg += sh.test('yes', 'Password')
    if msg.find('Password') >= 0:
        msg += sh.test('Huawei12#$', ':~', timeout)
    sh.close()
    print("return msg is %s" % msg)
    return msg

def host(host, cmd, expect, timeout=30, pwd=""):
    sh = SSH()
    sh.init(host, 'root', 'root')
    print("ssh host success" )
    msg = sh.test(cmd, expect, timeout)
    if msg.find('yes/no') >= 0:
        msg += sh.test('yes', 'password')
    if msg.find('password') >= 0:
        msg += sh.test(pwd, ':~', timeout)
    sh.close()
    return msg

if __name__ == '__main__':
    print('usage: \'python3 ssh.py host_ip\'')
    host =  sys.argv[1]
    host_path = sys.argv[2]
    device_path = sys.argv[3]
    scp_type = sys.argv[4]
    device_ip = sys.argv[5]
    file = '**'

    print("host_ip: %s" %host)
    print("host_path: %s" %host_path)
    print("device_path: %s" %device_path)
    print("scp_type: %s" %scp_type)

    sh = SSH()
    sh.init(host, 'root', 'root')
    msg = sh.test('ssh-keygen -f "/root/.ssh/known_hosts" -R %s' % device_ip, ':~')
    print(msg)
    if scp_type == 'device_to_host':
        print("copy from device to host") 
        msg = sh.test('scp -r root@%s:%s %s' %(device_ip, device_path, host_path), ':~', 30)
    else:
        print("copy from host to device")
        msg = sh.test('scp -r %s root@%s:%s' %(host_path, device_ip, device_path), 'yes/no', 30)
    
    if msg.find('yes/no') >= 0:
        msg += sh.test('yes', 'Password')
    if msg.find('Password') >= 0:
        msg += sh.test('Huawei12#$', ':~', 300) 
    print(msg)
    sh.close()  
    
