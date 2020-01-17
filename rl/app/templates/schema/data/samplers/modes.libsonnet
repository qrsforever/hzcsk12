// @file modes.libsonnet
// @brief
// @author QRS
// @version 1.0
// @date 2020-01-17 15:20

{
    _id_: '_k12.sampler.mode',
    name: { en: 'Mode', cn: self.en },
    type: 'string-enum',
    objs: [
        {
            name: { en: 'serial', cn: self.en },
            value: 'SerialSampler',
        },
        {
            name: { en: 'parallel cpu', cn: self.en },
            value: 'CpuSampler',
        },
        {
            name: { en: 'parallel gpu', cn: self.en },
            value: 'GpuSampler',
        },
        {
            name: { en: 'parallel alternating', cn: self.en },
            value: 'AlternatingSampler',
        },
        {
            name: { en: 'async serial', cn: self.en },
            value: 'AsyncSerialSampler',
        },
        {
            name: { en: 'async cpu', cn: self.en },
            value: 'AsyncCpuSampler',
        },
        {
            name: { en: 'async gpu', cn: self.en },
            value: 'AsyncGpuSampler',
        },
        {
            name: { en: 'async alternating', cn: self.en },
            value: 'AsyncAlternatingSampler',
        },
    ],
    default: 'CpuSampler',
}
