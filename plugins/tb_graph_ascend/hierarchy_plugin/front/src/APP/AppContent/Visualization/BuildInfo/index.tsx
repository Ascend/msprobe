/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { Button, Checkbox, Form, Input, InputNumber, message, Select, Steps } from 'antd';
import styles from './index.module.less';
import { useEffect } from 'react';
import { useVisualizedStore } from '../../../../store/useVisualizedStore';
import { escapeHTML } from '../../../../common/utils';
import type { ConvertParamsType } from '../../../../store/types/useVisualizedStore';
import { requestConvertToGraph } from '../../../../api/board';
import { useWatch } from 'antd/es/form/Form';
import { BUILD_STEP } from '../../../../common/constant';
import { useTranslation } from 'react-i18next';

const BuildInfo = () => {
  const {
    npuPathItems,
    benchPathItems,
    layerMappingItems,
    convertedGraphArgs,
    setConvertedGraphArgs,
    setCurrentBuildStep,
    fetchConvertedGraphData,
  } = useVisualizedStore();
  const [form] = Form.useForm();
  const { t } = useTranslation();
  const benchPathValue = useWatch('bench_path', form);
  const isParallelMerge = useWatch('is_parallel_merge', form);
  const [messageApi, contextHolder] = message.useMessage();
  useEffect(() => {
    fetchConvertedGraphData(messageApi);
  }, []);
  const onFinish = async (values: ConvertParamsType) => {
    const { success, error } = await requestConvertToGraph(values);
    if (success) {
      setConvertedGraphArgs(values);
      setCurrentBuildStep(BUILD_STEP.BUILD_PROGRESS);
    } else {
      messageApi.error(error);
    }
  };
  return (
    <div className={styles.buildInfoContainer}>
      {contextHolder}
      <div className={styles.buildWrapper}>
        <div className={styles.buildHeader}>
          <div className={styles.topContentWarning}>
            <span className={styles.topContentWarningImportant}>{t('build_info_desc_1')} (.vis.db/.vis)</span>
            <span>, {t('build_info_desc_2')}</span>
          </div>
          <p className={styles.topContentCenterTitle}>{t('build_info_desc_3')}:</p>
          <Steps
            progressDot
            direction="vertical"
            current={2}
            items={[
              {
                title: t('step1_title'),
                description: (
                  <div>
                    {t('step1_desc')}
                    <br />
                    <a
                      href="https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/pytorch_data_dump_instruct.md"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {t('pytorch_link_text')}
                    </a>
                    <a
                      href="https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/mindspore_data_dump_instruct.md"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ marginLeft: 20 }}
                    >
                      {t('mindspore_link_text')}
                    </a>
                  </div>
                ),
              },
              {
                title: t('step2_title'),
                description: t('step2_desc'),
              },
            ]}
          />
        </div>
        <div className="buildContent">
          <div className={styles.mainTitle}>{t('build_info_main_title')}</div>
          <div className={styles.mainTitleDesc}>
            {t('build_info_sub_title')}
            <a
              href="https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_visualization_instruct.md"
              target="_blank"
              rel="noopener noreferrer"
            >
              {t('build_info_pytorch_link_text')}
            </a>

            <a
              href="https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/mindspore_visualization_instruct.md"
              target="_blank"
              rel="noopener noreferrer"
              style={{ marginLeft: 10 }}
            >
              {t('build_info_mindspore_link_text')}
            </a>
          </div>
          <div className={styles.mainContent}>
            <Form
              name="layout-multiple-horizontal"
              layout="vertical"
              form={form}
              initialValues={convertedGraphArgs}
              onFinish={onFinish}
            >
              <div className={styles.subTitle}>{t('file_param_config')}</div>
              <Form.Item label={t('label_npu_path')} name="npu_path" rules={[{ required: true }]}>
                <Select allowClear options={npuPathItems} showSearch placeholder={t('placeholder_select')} />
              </Form.Item>
              <Form.Item label={t('label_bench_path')} name="bench_path">
                <Select
                  allowClear
                  options={benchPathItems}
                  showSearch
                  placeholder={t('placeholder_select')}
                  onChange={(value) => {
                    if (value) {
                      form.setFieldsValue({
                        output_path: `compare_${new Date().getTime()}`,
                      });
                    } else {
                      form.setFieldsValue({
                        output_path: `build_${new Date().getTime()}`,
                      });
                    }
                  }}
                />
              </Form.Item>
              <Form.Item
                label={t('label_output_path')}
                name="output_path"
                rules={[{ required: true }]}
                initialValue={benchPathValue ? `compare_${new Date().getTime()}` : `build_${new Date().getTime()}`}
              >
                <Input placeholder={t('placeholder_input')} />
              </Form.Item>
              <Form.Item name="is_print_compare_log" valuePropName="checked" className={styles.itemStyle}>
                <Checkbox>{t('checkbox_print_log')}</Checkbox>
              </Form.Item>
              <Form.Item name="is_parallel_merge" valuePropName="checked">
                <Checkbox>{t('checkbox_parallel_merge')}</Checkbox>
              </Form.Item>
              {isParallelMerge && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <div className={styles.parallelMergeWrapper}>
                    <div className={styles.parallelMergeTitle}>{t('debug_side')}</div>
                    <Form.Item
                      label={t('label_rank_size_npu')}
                      name={['parallel_merge', 'npu', 'rank_size']}
                      rules={[{ required: true }]}
                      className={styles.itemStyle}
                    >
                      <InputNumber controls min={1} style={{ width: '100%' }} />
                    </Form.Item>
                    <Form.Item
                      label={t('label_tp_npu')}
                      name={['parallel_merge', 'npu', 'tp']}
                      rules={[{ required: true }]}
                      className={styles.itemStyle}
                    >
                      <InputNumber controls min={1} style={{ width: '100%' }} />
                    </Form.Item>
                    <Form.Item
                      label={t('label_pp_npu')}
                      name={['parallel_merge', 'npu', 'pp']}
                      rules={[{ required: true }]}
                      className={styles.itemStyle}
                    >
                      <InputNumber controls min={1} style={{ width: '100%' }} />
                    </Form.Item>
                    <Form.Item
                      label={t('label_vpp_npu')}
                      name={['parallel_merge', 'npu', 'vpp']}
                      className={styles.itemStyle}
                    >
                      <InputNumber controls min={1} style={{ width: '100%' }} />
                    </Form.Item>
                    <Form.Item
                      label={t('label_order_npu')}
                      name={['parallel_merge', 'npu', 'order']}
                      normalize={(value) => escapeHTML(value)}
                    >
                      <Input placeholder={t('placeholder_input')} />
                    </Form.Item>
                  </div>
                  {benchPathValue && (
                    <div className={styles.parallelMergeWrapper}>
                      <div className={styles.parallelMergeTitle}>{t('benchmark_side')}</div>
                      <Form.Item
                        label={t('label_rank_size_bench')}
                        name={['parallel_merge', 'bench', 'rank_size']}
                        rules={[{ required: true }]}
                        className={styles.itemStyle}
                      >
                        <InputNumber controls min={1} style={{ width: '100%' }} />
                      </Form.Item>
                      <Form.Item
                        label={t('label_tp_bench')}
                        name={['parallel_merge', 'bench', 'tp']}
                        rules={[{ required: true }]}
                        className={styles.itemStyle}
                      >
                        <InputNumber controls min={1} style={{ width: '100%' }} />
                      </Form.Item>
                      <Form.Item
                        label={t('label_pp_bench')}
                        name={['parallel_merge', 'bench', 'pp']}
                        rules={[{ required: true }]}
                        className={styles.itemStyle}
                      >
                        <InputNumber controls min={1} style={{ width: '100%' }} />
                      </Form.Item>
                      <Form.Item
                        label={t('label_vpp_bench')}
                        name={['parallel_merge', 'bench', 'vpp']}
                        className={styles.itemStyle}
                      >
                        <InputNumber controls min={1} style={{ width: '100%' }} />
                      </Form.Item>
                      <Form.Item
                        label={t('label_order_bench')}
                        name={['parallel_merge', 'bench', 'order']}
                        normalize={(value) => escapeHTML(value)}
                      >
                        <Input placeholder={t('placeholder_input')} />
                      </Form.Item>
                    </div>
                  )}
                </div>
              )}
              <div className={styles.subTitle}>{t('more_options')}</div>
              <Form.Item label={t('label_layer_mapping')} name="layer_mapping">
                <Select allowClear options={layerMappingItems} placeholder={t('placeholder_select')} />
              </Form.Item>
              <Form.Item name="overflow_check" valuePropName="checked" className={styles.itemStyle}>
                <Checkbox>{t('checkbox_overflow_check')}</Checkbox>
              </Form.Item>
              <Form.Item name="fuzzy_match" valuePropName="checked">
                <Checkbox>{t('checkbox_fuzzy_match')}</Checkbox>
              </Form.Item>
              <Form.Item>
                <Button block type="primary" htmlType="submit">
                  {t('button_start_conversion')}
                </Button>
              </Form.Item>
            </Form>
            <div style={{ height: 100 }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};
export default BuildInfo;
