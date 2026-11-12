// 设备内小M路由原型：只放行已测形状，不改变外部张量和模型状态。
#ifndef A2O_HTP_SMALL_M_POLICY_H
#define A2O_HTP_SMALL_M_POLICY_H

// 0=原门槛；1=实验强制（含负对照）；2=保守形状白名单。
#ifndef A2O_HTP_SMALL_M_POLICY
#define A2O_HTP_SMALL_M_POLICY 2
#endif
#if A2O_HTP_SMALL_M_POLICY < 0 || A2O_HTP_SMALL_M_POLICY > 2
#error invalid A2O_HTP_SMALL_M_POLICY
#endif

static inline int a2o_htp_small_m_allowed(int m, int k, int n, int is_q4, int is_batched) {
    if (!is_q4 || is_batched || m != 31) return 0;
#if A2O_HTP_SMALL_M_POLICY == 1
    return 1;
#elif A2O_HTP_SMALL_M_POLICY == 2
    // 本机Qwen3.5 layer0 QKV/FFN Up；FFN Down仍用HVX。
    return k == 4096 && (n == 8192 || n == 12288);
#else
    (void) k; (void) n;
    return 0;
#endif
}
#endif
