GBPUSD, GBPEUR = 1.35, 1.17
TREASURY, XERO_BANK = 1_300_000.0, 159_456.76
wilson, vat = 25_894.92, 8_000.0
eris_owed = 11_463.98 + 14_655.89
iuk_per_q, interest_pa = 500_000.0/6, 0.0365
UPLIFT, CREDIT_RATE = 0.86, 0.145      # ERIS: 86% enhanced deduction, 14.5% payable credit
RD_APPORTION = 0.85

def emp(s): return (s + max(0.0, s-5000)*0.15 + s*0.03)/12
founders, core60 = 2*emp(50_000), 3*emp(60_000)
fa, rs, interns = emp(60_000), emp(75_000), 2*emp(30_000)
polaris, adarsh = 7175*1.20/GBPEUR, 70_000/GBPUSD/12
overhead, software = 2000.00+667.67+129.71+80.82+8.50, 500.0
node_gross, vast_inc, inference = 4375.00*2*0.75, 1635.0, 3000.0
azure, aws, google = 150_000/GBPUSD, 100_000/GBPUSD, 10_000.0

uk_full = founders + core60 + fa + rs
qual_mo = uk_full*RD_APPORTION + software + inference + node_gross      # 100% of node as R&D
opex_mo = uk_full + polaris + adarsh + overhead + software + node_gross + inference

Q  = qual_mo*12
LOSS = opex_mo*12 - vast_inc*12 + UPLIFT*Q      # trading loss incl. the 86% enhancement
CAP  = (1 + UPLIFT) * Q                          # credit is limited to 186% of qualifying
print("DOES THE VAST INCOME REDUCE THE ERIS CREDIT?")
print(f"  qualifying R&D expenditure      GBP {Q:>12,.0f} / yr")
print(f"  186% cap on surrenderable loss  GBP {CAP:>12,.0f}")
print(f"  actual surrenderable loss       GBP {LOSS:>12,.0f}")
print(f"  binding constraint              {'the 186% CAP' if LOSS > CAP else 'the LOSS'}")
credit = CREDIT_RATE * min(LOSS, CAP)
print(f"  payable credit                  GBP {credit:>12,.0f} / yr  "
      f"= {credit/Q:.2%} of qualifying")
headroom = LOSS - CAP
print(f"  loss headroom above the cap     GBP {headroom:>12,.0f}")
print(f"  Vast income                     GBP {vast_inc*12:>12,.0f} / yr")
print(f"  -> the income is {headroom/(vast_inc*12):.0f}x smaller than the headroom, so it does NOT")
print(f"     touch the credit. Vast income is {vast_inc*12/opex_mo/12:.1%} of total expenditure.\n")

def run(mode, extra, start, months=140):
    cash = start - wilson + vat + eris_owed
    az, aw, go = azure, aws, google
    q = 0.0
    for m in range(1, months+1):
        uk = founders + core60 + extra*emp(60_000)
        if m >= 2: uk += fa
        if m >= 3: uk += rs
        if 4 <= m <= 9: uk += interns
        burn = uk + polaris + adarsh + overhead + software
        cloud_q = inference
        if mode:
            claim, vinc = mode
            burn += node_gross - vinc
            cloud_q += node_gross*claim
        paid = inference
        for pot in range(3):
            if paid <= 0: break
            if pot == 0 and m <= 24 and az > 0:
                u = min(az, paid); az -= u; paid -= u
            elif pot == 1 and go > 0:
                u = min(go, paid); go -= u; paid -= u
            elif pot == 2 and aw > 0:
                u = min(aw, paid); aw -= u; paid -= u
        burn += paid
        q += uk*RD_APPORTION + software + cloud_q
        cash += (cash*interest_pa/12 if cash > 0 else 0)
        cash += iuk_per_q if (m % 3 == 0 and m <= 18) else 0
        if m % 12 == 0: cash += q*(1+UPLIFT)*CREDIT_RATE; q = 0.0
        cash -= burn
        if cash < 0:
            y, mo = 2026 + (8+m)//12, (8+m)%12 + 1
            return m, f"{y}-{mo:02d}"
    return None, ">11yr"

net = node_gross - vast_inc - node_gross*(1+UPLIFT)*CREDIT_RATE
print("TRUE COST OF THE TWO NODES, treating Vast as incidental income")
print(f"  gross rental (2x B7, 25% off)   GBP {node_gross:>9,.0f}")
print(f"  less Vast incidental income     GBP {-vast_inc:>9,.0f}")
print(f"  less ERIS on 100% of the cost   GBP {-node_gross*(1+UPLIFT)*CREDIT_RATE:>9,.0f}")
print(f"  NET                             GBP {net:>9,.0f} / month  = GBP {net*12:,.0f} / yr\n")

print("="*72)
print("RUNWAY, no revenue (GBP 1.30M treasury + 500k Innovate UK + ERIS)")
print(f"{'extra heads':>11} {'no nodes':>16} {'nodes, 100% claim':>20} {'cost':>8}")
for extra in (0, 2, 4):
    m0, d0 = run(None, extra, TREASURY)
    m1, d1 = run((1.00, vast_inc), extra, TREASURY)
    print(f"{extra:>11} {d0:>16} {d1:>20} {f'{m0-m1} mo':>8}")