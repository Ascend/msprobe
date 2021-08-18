# -*- coding: UTF-8 -*-
import os
import shutil
import datetime
from absl import app
from absl import flags
import tools.loss_check as loss_check
import tools.ckpt_check as ckpt_check
import tools.parse_trace as parse_trace
import tools.parse_aicpu as parse_aicpu
import tools.parse_hwts as parse_hwts
import tools.parse_runtime as parse_runtime
import tools.parse_aicore as parse_aicore
import tools.parse_op as parse_op
import tools.parse_info as parse_info
import tools.gen_analyze as gen_analyze

"""
@author: 
    w00282991
@email:
    wangbei5@huawei.com
@function:
    1. check loss fit.
    2. check ckpt consistency.
    3. analysis training trace.
    4. analysis aicpu op.
    5. analysis aicore op.
@usage:
    python3 training_analysis.py \
    --device_list=0 \
    --enable_loss_check=False \
    --loss_keyword='Average Loss' \
    --keyword_position=20 \
    --enable_ckpt_check=False \
    --enable_trace=False \
    --reduce_nums=0 \
    --enable_aicpu=False \
    --start_aicpu_name='bert/embeddings/DropOutGenMask' \
    --enable_hwts=True \
    --enable_pmu=True \
    --enable_statistics=False \
    --enable_op_info=False \
    --end_op_name='FlowCtrl_LoopCond_ASSIGNADD'
@directory structure:
    --training_analysis.py
    --tools
        --ckpt_check.py
        --loss_check.py
        --parse_aicore.py
        --parse_aicpu.py
        --parse_hwts.py
        --parse_op.py
        --parse_runtime.py
        --parse_trace.py
        --gen_analyze.py
    --result
        --exec-${timestamp}
            --0
                --ge_proto_xxxxx_Build.txt
            --train_${device_id}.log
            --npu
                --profiling
                    --JOBXXXXXXXXXXXXXXXXXX
                    --JOBYYYYYYYYYYYYYYYYYY
                    --JOBZZZZZZZZZZZZZZZZZZ
"""

FLAGS = flags.FLAGS
flags.DEFINE_list('device_list', [0],
                  'The device list which need to be analysis.')
flags.DEFINE_bool('enable_loss_check', False,
                  'Whether enable check loss fit.')
flags.DEFINE_string('loss_keyword', 'Average Loss',
                    'Keyword in train log to filter.')
flags.DEFINE_integer('keyword_position', 20,
                     'Keyword position in the sentence.')
flags.DEFINE_bool('enable_ckpt_check', False,
                  'Whether enable check ckpt consistency.')
flags.DEFINE_bool('enable_trace', True,
                  'Whether enable analysis trace log.')
flags.DEFINE_integer('reduce_nums', 0,
                     'Reduce slices the distributed network used, default 2 reduce slices, single node set to 0.')
flags.DEFINE_bool('enable_aicpu', True,
                  'Whether enable analysis aicpu log.')
flags.DEFINE_string('start_aicpu_name', 'IteratorV2',
                    'The end op name of every step.')
flags.DEFINE_bool('enable_hwts', True,
                  'Whether enable analysis hwts log.')
flags.DEFINE_bool('enable_pmu', False,
                  'Whether enable analysis pmu events.')
flags.DEFINE_bool('enable_statistics', True,
                  'Whether enable statistics op performance and timeline file.')
flags.DEFINE_bool('enable_op_info', False,
                  'Whether enable get op info.')
flags.DEFINE_string('end_op_name', 'FlowCtrl_LoopCond_ASSIGNADD',
                    'The end op name of every step.')


def print_log(data=None, level='INFO'):
    print("[%s] [%s] %s" % (datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S"), level, data))


def main(argv):
    del argv
    device_list = list(map(int, FLAGS.device_list))
    cur_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cur_path = os.path.abspath(os.path.dirname(__file__))
    result_list = os.listdir(os.path.join(cur_path, 'result'))
    result_list.sort(key=lambda fn: os.path.getmtime(os.path.join(cur_path, 'result') + "/" + fn))
    newest_result = os.path.join(os.path.join(cur_path, 'result'), result_list[-1])
    profiling_path = os.path.join(newest_result, 'npu/profiling/')
    # check loss fit
    if FLAGS.enable_loss_check:
        benchmark_loss_file = os.path.join(cur_path, 'benchmark_loss.npy')
        for device_id in device_list:
            print_log('Start to compare loss with benchmark from gpu for device: %s.' % device_id)
            loss_file = os.path.join(newest_result, 'train_%s.log' % device_id)
            loss_format = {'keyword': FLAGS.loss_keyword, 'position': FLAGS.keyword_position}
            threshold = 0.999
            result_file = os.path.join(cur_path, '%s_loss_trend_tag.%s.xlsx' % (cur_time, device_id))
            loss_check_result = loss_check.loss_check(benchmark_loss_file, loss_file,
                                                      loss_format, threshold, result_file)
            if loss_check_result == 1:
                print_log('Training file is not found in path: %s.' % loss_file, 'ERROR')
            elif loss_check_result == 2:
                print_log('Loss value is not number, please check loss format.', 'ERROR')
            elif loss_check_result == 3:
                print_log('Loss cosine similarity is not satisfied threshold: %s.' % threshold, 'ERROR')
            else:
                print_log('Loss cosine similarity is satisfied threshold: %s.' % threshold)
    # check ckpt consistency.
    if FLAGS.enable_ckpt_check:
        if len(device_list) == 1:
            print_log('Single device no need to check ckpt consistency.')
        else:
            print_log('Start to analysis ckpt consistency.')
            ckpt_check_result = ckpt_check.ckpt_check(newest_result)
            if ckpt_check_result == 1:
                print_log('Ckpt files numbers are not equal.', 'ERROR')
            elif ckpt_check_result == 2:
                print_log('Ckpt files are not the same', 'ERROR')
            else:
                print_log('Ckpt files are the same.')
    # analysis training trace.
    temp_trace_path = os.path.join(cur_path, '%s_training_trace' % cur_time)
    os.mkdir(temp_trace_path)
    if FLAGS.enable_trace:
        for device_id in device_list:
            print_log('Start to analysis training trace for device: %s.' % device_id)
            training_trace_list = list()
            result_file = os.path.join(cur_path, '%s_training_trace_tag.%s.xlsx' % (cur_time, device_id))
            for root, dirs, files in os.walk(profiling_path):
                for file in files:
                    if 'training_trace.46' in file and 'tag.' + str(device_id) in file and 'done' not in file:
                        training_trace_list.append(os.path.join(root, file))
            null_flag = False
            if training_trace_list:
                for file in training_trace_list:
                    if not os.path.isfile(file):
                        null_flag = True
            else:
                null_flag = True
            if not null_flag:
                parse_trace.parse_trace(training_trace_list, temp_trace_path, device_id, FLAGS.reduce_nums, result_file)
            else:
                print_log('Trace log for device %s is not exist.' % device_id, level='ERROR')
                break
    # analysis aicpu op.
    if FLAGS.enable_aicpu:
        for device_id in device_list:
            print_log('Start to analysis aicpu op for device: %s.' % device_id)
            aicpu_list = list()
            result_file = os.path.join(cur_path, '%s_aicpu_tag.%s.xlsx' % (cur_time, device_id))
            for root, dirs, files in os.walk(profiling_path):
                for file in files:
                    if 'AICPU.%s' % device_id in file and 'done' not in file:
                        aicpu_list.append(os.path.join(root, file))
            null_flag = False
            if aicpu_list:
                for file in aicpu_list:
                    if not os.path.isfile(file):
                        null_flag = True
            else:
                null_flag = True
            if not null_flag:
                parse_aicpu.parse_aicpu(aicpu_list, result_file, FLAGS.start_aicpu_name)
            else:
                print_log('Aicpu log for device %s is not exist.' % device_id, level='ERROR')
                break
    # analysis aicore op.
    temp_op_info_path = os.path.join(cur_path, '%s_op_info' % cur_time)
    os.mkdir(temp_op_info_path)
    temp_hwts_path = os.path.join(cur_path, '%s_hwts' % cur_time)
    os.mkdir(temp_hwts_path)
    temp_runtime_path = os.path.join(cur_path, '%s_runtime' % cur_time)
    os.mkdir(temp_runtime_path)
    temp_aicore_path = os.path.join(cur_path, '%s_aicore' % cur_time)
    os.mkdir(temp_aicore_path)
    if FLAGS.enable_hwts:
        benchmark_op_file = os.path.join(cur_path, 'benchmark_op.csv')
        for device_id in device_list:
            print_log('Start to analysis aicore op for device: %s.' % device_id)
            result_file = os.path.join(cur_path, '%s_aicore_tag.%s.xlsx' % (cur_time, device_id))
            statistics_file = os.path.join(cur_path, '%s_aicore_statistics_tag.%s.xlsx' % (cur_time, device_id))
            hwts_list = list()
            hwts_file = os.path.join(temp_hwts_path, 'parsed_hwts.%s.log' % device_id)
            for root, dirs, files in os.walk(profiling_path):
                for file in files:
                    if 'hwts.log.data.45' in file and 'tag.' + str(device_id) in file and 'done' not in file:
                        hwts_list.append(os.path.join(root, file))
            null_flag = False
            print(hwts_list)
            if hwts_list:
                for file in hwts_list:
                    if not os.path.isfile(file):
                        null_flag = True
            else:
                null_flag = True
            if not null_flag:
                parse_hwts.parse_hwts(hwts_list, temp_hwts_path, device_id, hwts_file)
            else:
                print_log('Hwts log for device %s is not exist.' % device_id, level='ERROR')
                break
            runtime_list = list()
            runtime_file = os.path.join(temp_runtime_path, 'parsed_runtime.%s.log' % device_id)
            for root, dirs, files in os.walk(profiling_path):
                for file in files:
                    if 'runtime.host.runtime.%s' % device_id in file and 'done' not in file:
                        runtime_list.append(os.path.join(root, file))
            null_flag = False
            if runtime_list:
                for file in runtime_list:
                    if not os.path.isfile(file):
                        null_flag = True
            else:
                null_flag = True
            if not null_flag:
                parse_runtime.parse_runtime(runtime_list, temp_runtime_path, device_id, runtime_file)
            else:
                print_log('Runtime log for device %s is not exist.' % device_id, level='ERROR')
                break
            aicore_file = None
            if FLAGS.enable_pmu:
                aicore_list = list()
                aicore_file = os.path.join(temp_aicore_path, 'parsed_aicore.%s.log' % device_id)
                for root, dirs, files in os.walk(profiling_path):
                    for file in files:
                        if 'aicore' in file and 'tag.' + str(device_id) in file and 'done' not in file:
                            aicore_list.append(os.path.join(root, file))
                null_flag = False
                if aicore_list:
                    for file in aicore_list:
                        if not os.path.isfile(file):
                            null_flag = True
                else:
                    null_flag = True
                if not null_flag:
                    parse_aicore.parse_aicore(aicore_list, temp_aicore_path, device_id, aicore_file)
                else:
                    print_log('Aicore log for device %s is not exist.' % device_id, level='ERROR')
                    break
            if FLAGS.end_op_name:
                op_info_file = None
                if FLAGS.enable_op_info:
                    ge_graph_path = os.path.join(newest_result, '%s' % device_id)
                    op_info_file = os.path.join(temp_op_info_path, 'parsed_op_info_tag.%s.log' % device_id)
                    ge_graph_list = list()
                    for root, dirs, files in os.walk(ge_graph_path):
                        for file in files:
                            if 'ge_proto' in file and 'Build.txt' in file:
                                ge_graph_list.append(os.path.join(root, file))
                    ge_graph_list.sort()
                    last_ge_graph = ge_graph_list[-1]
                    parse_info.parse_info(last_ge_graph, op_info_file)
                parse_op.parse_op(hwts_file, runtime_file, aicore_file, FLAGS.end_op_name, FLAGS.enable_pmu,
                                  FLAGS.enable_statistics, FLAGS.enable_op_info, op_info_file,
                                  result_file, statistics_file, benchmark_op_file)
            if FLAGS.enable_statistics:
                timestamp_trace_file = os.path.join(temp_trace_path, 'parsed_trace_timestamp.%s.xlsx' % device_id)
                gen_analyze.gen_analyze(cur_time, device_id, result_file, timestamp_trace_file,
                                        os.path.join(cur_path, '%s_aicpu_tag.%s.xlsx' % (cur_time, device_id)),
                                        FLAGS.enable_aicpu, FLAGS.enable_op_info)
    analysis_result = list()
    for root, dirs, files in os.walk(cur_path):
        for fold in dirs:
            if cur_time in fold:
                analysis_result.append(os.path.join(root, fold))
        for file in files:
            if cur_time in file:
                analysis_result.append(os.path.join(root, file))
    result_path = os.path.join(cur_path, '%s_training_analysis_result' % cur_time)
    os.mkdir(result_path)
    for result in analysis_result:
        shutil.move(result, result_path)


if __name__ == '__main__':
    app.run(main)
