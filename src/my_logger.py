import getpass
import logging
import sys


# 用于将控制台所有输出保存至文件，需放在代码最上面
class Logger(object):
    def __init__(self, filename='all.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        # self.terminal.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + message)
        # self.log.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + message)
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(stream=sys.stdout)
logger = logging.getLogger()


# 定义MyLog类
class MyLog(object):
    # 类MyLog的构造函数
    def __init__(self):
        self.user = getpass.getuser()
        self.logger = logging.getLogger(self.user)
        self.logger.setLevel(logging.DEBUG)
        # 日志文件名
        self.logFile = sys.argv[0][0:-3] + '.log'   #print(sys.argv[0])   代表文件名 输出 mylog.py
        # self.formatter = logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s\r\n')
        self.formatter = logging.Formatter('%(asctime)-12s %(levelname)-10s %(message)-12s')

        # 日志显示到屏幕上并输出到日志文件内
        # 输出到日志文件
        self.logHand = logging.FileHandler(self.logFile, encoding='utf8')
        self.logHand.setFormatter(self.formatter)
        self.logHand.setLevel(logging.DEBUG)

        # 输出到屏幕
        self.logHandSt = logging.StreamHandler()
        self.logHandSt.setFormatter(self.formatter)
        self.logHandSt.setLevel(logging.DEBUG)

        # 添加两个Handler
        self.logger.addHandler(self.logHand)
        # self.logger.addHandler(self.logHandSt)

    # 日志的5个级别对应以下的5个函数
    def debug(self,msg):
        self.logger.debug(msg)

    def info(self,msg):
        self.logger.info(msg)

    def warning(self,msg):
        self.logger.warning(msg)

    def error(self,msg):
        self.logger.error(msg)

    def critical(self,msg):
        self.logger.critical(msg)