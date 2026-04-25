import { useMemo, useState } from 'react';
import {
  ArrowDown,
  CheckCircle2,
  Globe,
  PencilLine,
  ShoppingCart,
  Wallet,
  WalletCards,
} from 'lucide-react';
import { SCENARIO_PRESETS, type ScenarioId } from './scenarioPresets';
import FraudResultScreen, { type Decision, type Transaction } from './UserResult';
import { addTransactionToHistory } from './transactionHistoryUtils';
import { scoreTransaction, type ScoreTransactionPayload, type ScoreTransactionResponse } from './api';
import { localeOptions, t, type Locale } from './i18n';

type TxType = 'MERCHANT' | 'P2P' | 'CASH_IN' | 'CASH_OUT';
type Channel = 'APP' | 'WEB' | 'AGENT' | 'QR';
type CountryCode = 'SG' | 'MY' | 'ID' | 'TH' | 'PH' | 'VN';
type ConnectivityMode = 'online' | 'intermittent' | 'offline_buffered';

const COUNTRY_CURRENCY: Record<CountryCode, string> = {
  SG: 'SGD',
  MY: 'MYR',
  ID: 'IDR',
  TH: 'THB',
  PH: 'PHP',
  VN: 'VND',
};

const ASEAN_CURRENCIES = ['SGD', 'MYR', 'IDR', 'THB', 'PHP', 'VND'] as const;

type SandboxFormState = {
  formScenario: ScenarioId;
  userId: string;
  amount: string;
  txType: TxType;
  isCrossBorder: boolean;
  walletId: string;
  currency: string;
  merchantName: string;
  deviceRiskScore: string;
  ipRiskScore: string;
  locationRiskScore: string;
  deviceId: string;
  deviceSharedUsers24h: string;
  accountAgeDays: string;
  simChangeRecent: boolean;
  channel: Channel;
  cashFlowVelocity1h: string;
  p2pCounterparties24h: string;
  sourceCountry: CountryCode;
  destinationCountry: CountryCode;
  isAgentAssisted: boolean;
  connectivityMode: ConnectivityMode;
};

const formScenarioMeta: Array<{
  id: ScenarioId;
  labelKey: string;
  descriptionKey: string;
  Icon: typeof ShoppingCart;
}> = [
  {
    id: 'everyday_purchase',
    labelKey: 'scenario.everyday_purchase.label',
    descriptionKey: 'scenario.everyday_purchase.narrative',
    Icon: ShoppingCart,
  },
  {
    id: 'large_amount',
    labelKey: 'scenario.large_amount.label',
    descriptionKey: 'scenario.large_amount.narrative',
    Icon: WalletCards,
  },
  {
    id: 'agent_cash_out',
    labelKey: 'scenario.agent_cash_out.label',
    descriptionKey: 'scenario.agent_cash_out.narrative',
    Icon: WalletCards,
  },
  {
    id: 'cross_border',
    labelKey: 'scenario.cross_border.label',
    descriptionKey: 'scenario.cross_border.narrative',
    Icon: Globe,
  },
  {
    id: 'custom',
    labelKey: 'scenario.custom.label',
    descriptionKey: 'scenario.custom.narrative',
    Icon: PencilLine,
  },
];

const demoCopy: Record<Locale, Record<string, string>> = {
  en: {
    'demo.eyebrow': 'Integration sandbox',
    'demo.title': 'Live transaction sandbox',
    'demo.subtitle': "Pick a scenario preset, fill in the details, then tap to check if it's safe.",
    'demo.language': 'Accessibility language',
    'demo.section.advanced': 'Advanced context (support staff only)',
    'demo.section.advancedSubtitle': 'Auto-filled from selected preset. Adjust only when investigating customer-reported edge cases.',
    'demo.section.riskScores': 'Risk scores',
    'demo.section.deviceSignals': 'Device signals',
    'demo.section.behavior': 'Behavior velocity and channel',
    'demo.field.sourceCountry': 'Source country',
    'demo.field.destinationCountry': 'Destination country',
    'demo.field.connectivity': 'Connectivity',
    'demo.field.agentAssisted': 'Agent-assisted',
    'demo.field.deviceRisk': 'Device risk score',
    'demo.field.ipRisk': 'IP risk score',
    'demo.field.locationRisk': 'Location risk score',
    'demo.field.deviceId': 'Device ID',
    'demo.field.deviceShared': 'Device shared users (24h)',
    'demo.field.accountAge': 'Account age (days)',
    'demo.field.simChanged': 'SIM recently changed',
    'demo.field.channel': 'Channel',
    'demo.field.cashVelocity': 'Cash flow velocity (1h)',
    'demo.field.counterparties': 'P2P counterparties (24h)',
    'demo.phone.review': 'Review Transaction',
    'demo.phone.from': 'From account',
    'demo.phone.change': 'Change',
    'demo.phone.summary': 'Transaction summary',
    'demo.phone.corridor': 'Corridor',
    'demo.phone.total': 'Total',
    'demo.phone.pay': 'Pay Now',
    'demo.phone.processing': 'Processing',
    'demo.tx.peer': 'Peer transfer',
    'demo.tx.cashIn': 'Cash in',
    'demo.tx.cashOut': 'Cash out',
    'demo.tx.merchant': 'Merchant payment',
    'demo.connectivity.live': 'Live',
    'demo.connectivity.intermittent': 'Intermittent',
    'demo.connectivity.degraded': 'Degraded',
    'demo.status.calling': 'Calling fraud engine...',
    'demo.status.liveResponse': 'Live API response',
    'demo.status.offlineEstimate': 'Offline estimate',
    'demo.status.showingEstimate': 'showing local estimate',
    'demo.status.decisionSource': 'Decision source',
    'demo.status.normalisedAmount': 'Normalised amount',
    'demo.status.ringScore': 'Ring score',
    'demo.status.ringMatch': 'Ring match',
    'demo.status.accountRing': 'Account in ring',
    'demo.status.attributeMatch': 'Attribute match',
    'demo.status.latency': 'Latency',
    'demo.status.topDrivers': 'Top feature drivers',
    'demo.status.reasonCodes': 'Reason codes',
    'demo.status.live': 'Live',
    'demo.status.cached': 'Cached',
    'demo.status.degraded': 'Degraded',
    'demo.status.apiUnavailable': 'API unavailable',
  },
  id: {
    'demo.eyebrow': 'Sandbox integrasi',
    'demo.title': 'Sandbox transaksi langsung',
    'demo.subtitle': 'Pilih preset skenario, isi detailnya, lalu ketuk untuk mengecek apakah aman.',
    'demo.language': 'Bahasa aksesibilitas',
    'demo.section.advanced': 'Konteks lanjutan (khusus staf dukungan)',
    'demo.section.advancedSubtitle': 'Terisi otomatis dari preset yang dipilih. Ubah hanya saat menyelidiki kasus khusus yang dilaporkan pelanggan.',
    'demo.section.riskScores': 'Skor risiko',
    'demo.section.deviceSignals': 'Sinyal perangkat',
    'demo.section.behavior': 'Kecepatan perilaku dan kanal',
    'demo.field.sourceCountry': 'Negara asal',
    'demo.field.destinationCountry': 'Negara tujuan',
    'demo.field.connectivity': 'Konektivitas',
    'demo.field.agentAssisted': 'Dibantu agen',
    'demo.field.deviceRisk': 'Skor risiko perangkat',
    'demo.field.ipRisk': 'Skor risiko IP',
    'demo.field.locationRisk': 'Skor risiko lokasi',
    'demo.field.deviceId': 'ID perangkat',
    'demo.field.deviceShared': 'Pengguna berbagi perangkat (24 jam)',
    'demo.field.accountAge': 'Usia akun (hari)',
    'demo.field.simChanged': 'SIM baru saja diganti',
    'demo.field.channel': 'Kanal',
    'demo.field.cashVelocity': 'Kecepatan arus kas (1 jam)',
    'demo.field.counterparties': 'Lawan transaksi P2P (24 jam)',
    'demo.phone.review': 'Tinjau Transaksi',
    'demo.phone.from': 'Dari akun',
    'demo.phone.change': 'Ubah',
    'demo.phone.summary': 'Ringkasan transaksi',
    'demo.phone.corridor': 'Koridor',
    'demo.phone.total': 'Total',
    'demo.phone.pay': 'Bayar Sekarang',
    'demo.phone.processing': 'Memproses',
    'demo.tx.peer': 'Transfer ke orang',
    'demo.tx.cashIn': 'Setor tunai',
    'demo.tx.cashOut': 'Tarik tunai',
    'demo.tx.merchant': 'Pembayaran merchant',
    'demo.connectivity.live': 'Langsung',
    'demo.connectivity.intermittent': 'Tidak stabil',
    'demo.connectivity.degraded': 'Terbatas',
    'demo.status.calling': 'Memanggil mesin fraud...',
    'demo.status.liveResponse': 'Respons API langsung',
    'demo.status.offlineEstimate': 'Estimasi offline',
    'demo.status.showingEstimate': 'menampilkan estimasi lokal',
    'demo.status.decisionSource': 'Sumber keputusan',
    'demo.status.normalisedAmount': 'Nominal ternormalisasi',
    'demo.status.ringScore': 'Skor ring',
    'demo.status.ringMatch': 'Kecocokan ring',
    'demo.status.accountRing': 'Akun dalam ring',
    'demo.status.attributeMatch': 'Kecocokan atribut',
    'demo.status.latency': 'Latensi',
    'demo.status.topDrivers': 'Faktor fitur teratas',
    'demo.status.reasonCodes': 'Kode alasan',
    'demo.status.live': 'Langsung',
    'demo.status.cached': 'Tertembolok',
    'demo.status.degraded': 'Terbatas',
    'demo.status.apiUnavailable': 'API tidak tersedia',
    'scenario.agent_cash_out.label': 'Tarik tunai via agen',
    'scenario.agent_cash_out.narrative': 'Tarik tunai dengan bantuan agen dan perlindungan koneksi tidak stabil.',
  },
  th: {
    'demo.eyebrow': 'แซนด์บ็อกซ์การเชื่อมต่อ',
    'demo.title': 'แซนด์บ็อกซ์ธุรกรรมแบบสด',
    'demo.subtitle': 'เลือกสถานการณ์ กรอกรายละเอียด แล้วแตะเพื่อตรวจสอบว่าปลอดภัยหรือไม่',
    'demo.language': 'ภาษาสำหรับการเข้าถึง',
    'demo.section.advanced': 'บริบทขั้นสูง (สำหรับเจ้าหน้าที่สนับสนุนเท่านั้น)',
    'demo.section.advancedSubtitle': 'กรอกอัตโนมัติจากสถานการณ์ที่เลือก ปรับเฉพาะเมื่อตรวจสอบกรณีเฉพาะที่ลูกค้าแจ้ง',
    'demo.section.riskScores': 'คะแนนความเสี่ยง',
    'demo.section.deviceSignals': 'สัญญาณอุปกรณ์',
    'demo.section.behavior': 'ความเร็วพฤติกรรมและช่องทาง',
    'demo.field.sourceCountry': 'ประเทศต้นทาง',
    'demo.field.destinationCountry': 'ประเทศปลายทาง',
    'demo.field.connectivity': 'การเชื่อมต่อ',
    'demo.field.agentAssisted': 'มีเอเจนต์ช่วย',
    'demo.field.deviceRisk': 'คะแนนความเสี่ยงอุปกรณ์',
    'demo.field.ipRisk': 'คะแนนความเสี่ยง IP',
    'demo.field.locationRisk': 'คะแนนความเสี่ยงตำแหน่ง',
    'demo.field.deviceId': 'รหัสอุปกรณ์',
    'demo.field.deviceShared': 'ผู้ใช้ร่วมอุปกรณ์ (24 ชม.)',
    'demo.field.accountAge': 'อายุบัญชี (วัน)',
    'demo.field.simChanged': 'เพิ่งเปลี่ยนซิม',
    'demo.field.channel': 'ช่องทาง',
    'demo.field.cashVelocity': 'ความเร็วกระแสเงินสด (1 ชม.)',
    'demo.field.counterparties': 'คู่ธุรกรรม P2P (24 ชม.)',
    'demo.phone.review': 'ตรวจสอบธุรกรรม',
    'demo.phone.from': 'จากบัญชี',
    'demo.phone.change': 'เปลี่ยน',
    'demo.phone.summary': 'สรุปธุรกรรม',
    'demo.phone.corridor': 'เส้นทางธุรกรรม',
    'demo.phone.total': 'รวม',
    'demo.phone.pay': 'ชำระเงินตอนนี้',
    'demo.phone.processing': 'กำลังประมวลผล',
    'demo.tx.peer': 'โอนให้บุคคล',
    'demo.tx.cashIn': 'ฝากเงินสด',
    'demo.tx.cashOut': 'ถอนเงินสด',
    'demo.tx.merchant': 'ชำระร้านค้า',
    'demo.connectivity.live': 'สด',
    'demo.connectivity.intermittent': 'ไม่เสถียร',
    'demo.connectivity.degraded': 'จำกัด',
    'demo.status.calling': 'กำลังเรียกเครื่องมือตรวจจับ fraud...',
    'demo.status.liveResponse': 'คำตอบ API แบบสด',
    'demo.status.offlineEstimate': 'ประมาณการออฟไลน์',
    'demo.status.showingEstimate': 'แสดงประมาณการภายในเครื่อง',
    'demo.status.decisionSource': 'แหล่งที่มาของการตัดสินใจ',
    'demo.status.normalisedAmount': 'ยอดเงินที่ปรับมาตรฐาน',
    'demo.status.ringScore': 'คะแนนกลุ่มเชื่อมโยง',
    'demo.status.ringMatch': 'การจับคู่กลุ่มเชื่อมโยง',
    'demo.status.accountRing': 'บัญชีอยู่ในกลุ่มเชื่อมโยง',
    'demo.status.attributeMatch': 'ตรงกับแอตทริบิวต์',
    'demo.status.latency': 'เวลาแฝง',
    'demo.status.topDrivers': 'ปัจจัยสำคัญสูงสุด',
    'demo.status.reasonCodes': 'รหัสเหตุผล',
    'demo.status.live': 'สด',
    'demo.status.cached': 'แคช',
    'demo.status.degraded': 'จำกัด',
    'demo.status.apiUnavailable': 'API ไม่พร้อมใช้งาน',
    'scenario.agent_cash_out.label': 'ถอนเงินผ่านเอเจนต์',
    'scenario.agent_cash_out.narrative': 'ถอนเงินโดยมีเอเจนต์ช่วย พร้อมมาตรการป้องกันเมื่อการเชื่อมต่อไม่เสถียร',
  },
  vi: {
    'demo.eyebrow': 'Sandbox tích hợp',
    'demo.title': 'Sandbox giao dịch trực tiếp',
    'demo.subtitle': 'Chọn một kịch bản, nhập thông tin rồi chạm để kiểm tra giao dịch có an toàn không.',
    'demo.language': 'Ngôn ngữ hỗ trợ truy cập',
    'demo.section.advanced': 'Ngữ cảnh nâng cao (chỉ dành cho nhân viên hỗ trợ)',
    'demo.section.advancedSubtitle': 'Được tự động điền từ kịch bản đã chọn. Chỉ chỉnh khi điều tra các trường hợp đặc biệt do khách hàng báo cáo.',
    'demo.section.riskScores': 'Điểm rủi ro',
    'demo.section.deviceSignals': 'Tín hiệu thiết bị',
    'demo.section.behavior': 'Tốc độ hành vi và kênh',
    'demo.field.sourceCountry': 'Quốc gia nguồn',
    'demo.field.destinationCountry': 'Quốc gia đích',
    'demo.field.connectivity': 'Kết nối',
    'demo.field.agentAssisted': 'Có đại lý hỗ trợ',
    'demo.field.deviceRisk': 'Điểm rủi ro thiết bị',
    'demo.field.ipRisk': 'Điểm rủi ro IP',
    'demo.field.locationRisk': 'Điểm rủi ro vị trí',
    'demo.field.deviceId': 'ID thiết bị',
    'demo.field.deviceShared': 'Người dùng chung thiết bị (24h)',
    'demo.field.accountAge': 'Tuổi tài khoản (ngày)',
    'demo.field.simChanged': 'SIM vừa được thay đổi',
    'demo.field.channel': 'Kênh',
    'demo.field.cashVelocity': 'Tốc độ dòng tiền (1h)',
    'demo.field.counterparties': 'Đối tác P2P (24h)',
    'demo.phone.review': 'Xem lại giao dịch',
    'demo.phone.from': 'Từ tài khoản',
    'demo.phone.change': 'Thay đổi',
    'demo.phone.summary': 'Tóm tắt giao dịch',
    'demo.phone.corridor': 'Tuyến giao dịch',
    'demo.phone.total': 'Tổng cộng',
    'demo.phone.pay': 'Thanh toán ngay',
    'demo.phone.processing': 'Đang xử lý',
    'demo.tx.peer': 'Chuyển cho cá nhân',
    'demo.tx.cashIn': 'Nạp tiền mặt',
    'demo.tx.cashOut': 'Rút tiền mặt',
    'demo.tx.merchant': 'Thanh toán merchant',
    'demo.connectivity.live': 'Trực tiếp',
    'demo.connectivity.intermittent': 'Không ổn định',
    'demo.connectivity.degraded': 'Hạn chế',
    'demo.status.calling': 'Đang gọi fraud engine...',
    'demo.status.liveResponse': 'Phản hồi API trực tiếp',
    'demo.status.offlineEstimate': 'Ước tính ngoại tuyến',
    'demo.status.showingEstimate': 'đang hiển thị ước tính cục bộ',
    'demo.status.decisionSource': 'Nguồn quyết định',
    'demo.status.normalisedAmount': 'Số tiền đã chuẩn hóa',
    'demo.status.ringScore': 'Điểm vòng gian lận',
    'demo.status.ringMatch': 'Khớp vòng gian lận',
    'demo.status.accountRing': 'Tài khoản trong vòng',
    'demo.status.attributeMatch': 'Khớp thuộc tính',
    'demo.status.latency': 'Độ trễ',
    'demo.status.topDrivers': 'Yếu tố ảnh hưởng chính',
    'demo.status.reasonCodes': 'Mã lý do',
    'demo.status.live': 'Trực tiếp',
    'demo.status.cached': 'Đã lưu cache',
    'demo.status.degraded': 'Hạn chế',
    'demo.status.apiUnavailable': 'API không khả dụng',
    'scenario.agent_cash_out.label': 'Rút tiền qua đại lý',
    'scenario.agent_cash_out.narrative': 'Rút tiền có đại lý hỗ trợ, kèm biện pháp bảo vệ khi kết nối không ổn định.',
  },
  tl: {
    'demo.eyebrow': 'Integration sandbox',
    'demo.title': 'Live transaction sandbox',
    'demo.subtitle': 'Pumili ng scenario preset, ilagay ang detalye, tapos i-tap para malaman kung ligtas.',
    'demo.language': 'Wika para sa accessibility',
    'demo.section.advanced': 'Advanced na konteksto (para lang sa support staff)',
    'demo.section.advancedSubtitle': 'Awtomatikong napupunan mula sa napiling preset. Baguhin lang kapag iniimbestigahan ang espesyal na kasong iniulat ng customer.',
    'demo.section.riskScores': 'Mga risk score',
    'demo.section.deviceSignals': 'Mga signal ng device',
    'demo.section.behavior': 'Bilis ng kilos at channel',
    'demo.field.sourceCountry': 'Pinanggalingang bansa',
    'demo.field.destinationCountry': 'Patutunguhang bansa',
    'demo.field.connectivity': 'Koneksyon',
    'demo.field.agentAssisted': 'Tinulungan ng agent',
    'demo.field.deviceRisk': 'Risk score ng device',
    'demo.field.ipRisk': 'Risk score ng IP',
    'demo.field.locationRisk': 'Risk score ng lokasyon',
    'demo.field.deviceId': 'Device ID',
    'demo.field.deviceShared': 'Mga gumagamit ng parehong device (24h)',
    'demo.field.accountAge': 'Edad ng account (araw)',
    'demo.field.simChanged': 'Kamakailan lang napalitan ang SIM',
    'demo.field.channel': 'Channel',
    'demo.field.cashVelocity': 'Bilis ng cash flow (1h)',
    'demo.field.counterparties': 'Mga P2P counterparties (24h)',
    'demo.phone.review': 'Suriin ang Transaksyon',
    'demo.phone.from': 'Mula sa account',
    'demo.phone.change': 'Palitan',
    'demo.phone.summary': 'Buod ng transaksyon',
    'demo.phone.corridor': 'Koridor',
    'demo.phone.total': 'Kabuuan',
    'demo.phone.pay': 'Magbayad Ngayon',
    'demo.phone.processing': 'Pinoproseso',
    'demo.tx.peer': 'Padala sa tao',
    'demo.tx.cashIn': 'Cash in',
    'demo.tx.cashOut': 'Cash out',
    'demo.tx.merchant': 'Bayad sa merchant',
    'demo.connectivity.live': 'Live',
    'demo.connectivity.intermittent': 'Paputol-putol',
    'demo.connectivity.degraded': 'Limitado',
    'demo.status.calling': 'Tinatawagan ang fraud engine...',
    'demo.status.liveResponse': 'Live API response',
    'demo.status.offlineEstimate': 'Offline estimate',
    'demo.status.showingEstimate': 'ipinapakita ang lokal na estimate',
    'demo.status.decisionSource': 'Pinagmulan ng desisyon',
    'demo.status.normalisedAmount': 'Na-normalize na halaga',
    'demo.status.ringScore': 'Ring score',
    'demo.status.ringMatch': 'Ring match',
    'demo.status.accountRing': 'Account sa ring',
    'demo.status.attributeMatch': 'Attribute match',
    'demo.status.latency': 'Latency',
    'demo.status.topDrivers': 'Pangunahing feature drivers',
    'demo.status.reasonCodes': 'Reason codes',
    'demo.status.live': 'Live',
    'demo.status.cached': 'Cached',
    'demo.status.degraded': 'Limitado',
    'demo.status.apiUnavailable': 'Hindi available ang API',
    'scenario.agent_cash_out.label': 'Cash-out sa agent',
    'scenario.agent_cash_out.narrative': 'Cash-out na tinulungan ng agent na may proteksyon kapag mahina ang koneksyon.',
  },
  ms: {
    'demo.eyebrow': 'Kotak pasir integrasi',
    'demo.title': 'Kotak pasir transaksi langsung',
    'demo.subtitle': 'Pilih preset senario, isi butiran, kemudian ketik untuk semak sama ada selamat.',
    'demo.language': 'Bahasa aksesibiliti',
    'demo.section.advanced': 'Konteks lanjutan (untuk staf sokongan sahaja)',
    'demo.section.advancedSubtitle': 'Diisi automatik daripada preset yang dipilih. Laraskan hanya apabila menyiasat kes khas yang dilaporkan pelanggan.',
    'demo.section.riskScores': 'Skor risiko',
    'demo.section.deviceSignals': 'Isyarat peranti',
    'demo.section.behavior': 'Kelajuan tingkah laku dan saluran',
    'demo.field.sourceCountry': 'Negara sumber',
    'demo.field.destinationCountry': 'Negara destinasi',
    'demo.field.connectivity': 'Ketersambungan',
    'demo.field.agentAssisted': 'Dibantu ejen',
    'demo.field.deviceRisk': 'Skor risiko peranti',
    'demo.field.ipRisk': 'Skor risiko IP',
    'demo.field.locationRisk': 'Skor risiko lokasi',
    'demo.field.deviceId': 'ID peranti',
    'demo.field.deviceShared': 'Pengguna berkongsi peranti (24j)',
    'demo.field.accountAge': 'Umur akaun (hari)',
    'demo.field.simChanged': 'SIM baru ditukar',
    'demo.field.channel': 'Saluran',
    'demo.field.cashVelocity': 'Kelajuan aliran tunai (1j)',
    'demo.field.counterparties': 'Pihak lawan P2P (24j)',
    'demo.phone.review': 'Semak Transaksi',
    'demo.phone.from': 'Daripada akaun',
    'demo.phone.change': 'Tukar',
    'demo.phone.summary': 'Ringkasan transaksi',
    'demo.phone.corridor': 'Koridor',
    'demo.phone.total': 'Jumlah',
    'demo.phone.pay': 'Bayar Sekarang',
    'demo.phone.processing': 'Memproses',
    'demo.tx.peer': 'Pindahan kepada individu',
    'demo.tx.cashIn': 'Tambah tunai',
    'demo.tx.cashOut': 'Keluarkan tunai',
    'demo.tx.merchant': 'Bayaran merchant',
    'demo.connectivity.live': 'Langsung',
    'demo.connectivity.intermittent': 'Tidak stabil',
    'demo.connectivity.degraded': 'Terhad',
    'demo.status.calling': 'Memanggil fraud engine...',
    'demo.status.liveResponse': 'Respons API langsung',
    'demo.status.offlineEstimate': 'Anggaran luar talian',
    'demo.status.showingEstimate': 'memaparkan anggaran setempat',
    'demo.status.decisionSource': 'Sumber keputusan',
    'demo.status.normalisedAmount': 'Jumlah dinormalkan',
    'demo.status.ringScore': 'Skor ring',
    'demo.status.ringMatch': 'Padanan ring',
    'demo.status.accountRing': 'Akaun dalam ring',
    'demo.status.attributeMatch': 'Padanan atribut',
    'demo.status.latency': 'Kependaman',
    'demo.status.topDrivers': 'Pemacu ciri utama',
    'demo.status.reasonCodes': 'Kod sebab',
    'demo.status.live': 'Langsung',
    'demo.status.cached': 'Cache',
    'demo.status.degraded': 'Terhad',
    'demo.status.apiUnavailable': 'API tidak tersedia',
    'scenario.agent_cash_out.label': 'Cash-out melalui ejen',
    'scenario.agent_cash_out.narrative': 'Cash-out dibantu ejen dengan perlindungan apabila sambungan tidak stabil.',
  },
};

const demoText = (locale: Locale, key: string): string => demoCopy[locale]?.[key] ?? demoCopy.en[key] ?? key;
const sharedText = (locale: Locale, key: string): string => {
  if (demoCopy[locale]?.[key]) {
    return demoCopy[locale][key];
  }
  const value = t(locale, key);
  return value === key ? demoText(locale, key) : value;
};

function createInitialFormState(): SandboxFormState {
  const basePreset = SCENARIO_PRESETS.everyday_purchase;

  return {
    formScenario: 'everyday_purchase',
    userId: basePreset.ui.user_id,
    amount: basePreset.ui.amount,
    txType: basePreset.ui.tx_type,
    isCrossBorder: basePreset.ui.is_cross_border,
    walletId: basePreset.ui.wallet_id,
    currency: basePreset.ui.currency,
    merchantName: basePreset.ui.merchant_name,
    deviceRiskScore: String(basePreset.risk.device_risk_score),
    ipRiskScore: String(basePreset.risk.ip_risk_score),
    locationRiskScore: String(basePreset.risk.location_risk_score),
    deviceId: basePreset.risk.device_id,
    deviceSharedUsers24h: String(basePreset.risk.device_shared_users_24h),
    accountAgeDays: String(basePreset.risk.account_age_days),
    simChangeRecent: basePreset.risk.sim_change_recent,
    channel: basePreset.risk.channel,
    cashFlowVelocity1h: String(basePreset.risk.cash_flow_velocity_1h),
    p2pCounterparties24h: String(basePreset.risk.p2p_counterparties_24h),
    sourceCountry: (basePreset.ui.source_country ?? 'ID') as CountryCode,
    destinationCountry: (basePreset.ui.destination_country ?? 'ID') as CountryCode,
    isAgentAssisted: basePreset.ui.is_agent_assisted ?? false,
    connectivityMode: (basePreset.ui.connectivity_mode ?? 'online') as ConnectivityMode,
  };
}

function formatAmountForPreview(value: string): string {
  const digitsOnly = value.replace(/[^\d]/g, '');
  const parsed = Number.parseInt(digitsOnly || '0', 10);
  return `Rp${new Intl.NumberFormat('id-ID').format(parsed)}`;
}

function inferWalletDisplayName(value: string): string {
  if (!value.trim()) return 'Main Pocket';
  if (value === 'user_1001' || value === 'wallet_1001') return 'Main Pocket';
  return value;
}

function inferRecipientName(value: string): string {
  if (!value.trim()) return 'Recipient';
  return value;
}

function applyScenarioPreset(current: SandboxFormState, scenarioId: ScenarioId): SandboxFormState {
  const preset = SCENARIO_PRESETS[scenarioId];

  if (scenarioId === 'custom') {
    return {
      ...current,
      formScenario: scenarioId,
      userId: '',
      amount: '',
      txType: 'MERCHANT',
      isCrossBorder: false,
      walletId: '',
      currency: '',
      merchantName: '',
      deviceRiskScore: String(preset.risk.device_risk_score),
      ipRiskScore: String(preset.risk.ip_risk_score),
      locationRiskScore: String(preset.risk.location_risk_score),
      deviceId: preset.risk.device_id,
      deviceSharedUsers24h: String(preset.risk.device_shared_users_24h),
      accountAgeDays: String(preset.risk.account_age_days),
      simChangeRecent: preset.risk.sim_change_recent,
      channel: preset.risk.channel,
      cashFlowVelocity1h: String(preset.risk.cash_flow_velocity_1h),
      p2pCounterparties24h: String(preset.risk.p2p_counterparties_24h),
      sourceCountry: 'SG' as CountryCode,
      destinationCountry: 'SG' as CountryCode,
      isAgentAssisted: false,
      connectivityMode: 'online' as ConnectivityMode,
    };
  }

  return {
    ...current,
    formScenario: scenarioId,
    userId: preset.ui.user_id,
    amount: preset.ui.amount,
    txType: preset.ui.tx_type,
    isCrossBorder: preset.ui.is_cross_border,
    walletId: preset.ui.wallet_id,
    currency: preset.ui.currency,
    merchantName: preset.ui.merchant_name,
    deviceRiskScore: String(preset.risk.device_risk_score),
    ipRiskScore: String(preset.risk.ip_risk_score),
    locationRiskScore: String(preset.risk.location_risk_score),
    deviceId: preset.risk.device_id,
    deviceSharedUsers24h: String(preset.risk.device_shared_users_24h),
    accountAgeDays: String(preset.risk.account_age_days),
    simChangeRecent: preset.risk.sim_change_recent,
    channel: preset.risk.channel,
    cashFlowVelocity1h: String(preset.risk.cash_flow_velocity_1h),
    p2pCounterparties24h: String(preset.risk.p2p_counterparties_24h),
    sourceCountry: (preset.ui.source_country ?? 'ID') as CountryCode,
    destinationCountry: (preset.ui.destination_country ?? 'ID') as CountryCode,
    isAgentAssisted: preset.ui.is_agent_assisted ?? false,
    connectivityMode: (preset.ui.connectivity_mode ?? 'online') as ConnectivityMode,
  };
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className="mb-2 block text-[12px] font-medium tracking-[-0.01em] text-[#CECFD2]">{children}</label>;
}

function FieldInput({
  value,
  onChange,
  placeholder,
  type = 'text',
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      className="w-full rounded-xl border border-white/10 bg-[#171B22] px-4 py-3 text-[14px] text-white outline-none transition focus:border-[#1273E7]"
    />
  );
}

function FieldSelect<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: Array<{ value: T; label: string }>;
}) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as T)}
        className="w-full appearance-none rounded-xl border border-white/10 bg-[#171B22] px-4 py-3 pr-10 text-[14px] text-white outline-none transition focus:border-[#1273E7]"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <div className="pointer-events-none absolute inset-y-0 right-3 flex items-center text-[#94979C]">
        <ArrowDown size={16} />
      </div>
    </div>
  );
}

function StatusBar() {
  return (
    <div className="flex h-[53px] items-end justify-between px-[18px] pb-[6px]">
      <div className="w-[54px] text-[17px] font-semibold tracking-[-0.4px] text-[#F3F6FB]">9:41</div>
      <div className="h-[37px] w-[125px] rounded-full bg-[#05070A]" />
      <div className="flex w-[72px] items-center justify-end gap-[8px]">
        <div className="flex items-end gap-[2px]">
          <span className="h-[5px] w-[3px] rounded-full bg-[#F3F6FB]" />
          <span className="h-[7px] w-[3px] rounded-full bg-[#F3F6FB]" />
          <span className="h-[9px] w-[3px] rounded-full bg-[#F3F6FB]" />
          <span className="h-[11px] w-[3px] rounded-full bg-[#F3F6FB]" />
        </div>
        <div className="relative h-[11px] w-[16px]">
          <div className="absolute inset-x-0 top-0 h-[11px] rounded-t-full border border-[#F3F6FB] border-b-0" />
          <div className="absolute inset-x-[2px] top-[4px] h-[5px] rounded-full bg-[#F3F6FB]" />
        </div>
        <div className="relative h-[13px] w-[27px] rounded-[4px] border border-[#F3F6FB]/45">
          <div className="absolute left-[1.5px] top-[1.5px] h-[8px] w-[19px] rounded-[2px] bg-[#F3F6FB]" />
          <div className="absolute right-[-3px] top-[4px] h-[4px] w-[2px] rounded-r bg-[#F3F6FB]/70" />
        </div>
      </div>
    </div>
  );
}

function RecipientAvatar() {
  return (
    <div className="relative h-[34px] w-[34px] overflow-hidden rounded-full bg-[#784CFC]">
      <div className="absolute left-[6px] top-[6px] h-[12px] w-[22px] rounded-full bg-[#FFB0B7]" />
      <div className="absolute -bottom-[2px] left-[4px] h-[18px] w-[26px] rounded-full bg-[#F3A4C0]" />
      <div className="absolute -left-[4px] top-[8px] h-[16px] w-[10px] rounded-r-full bg-[#5C28E2]" />
    </div>
  );
}

function HomeIndicator() {
  return <div className="mx-auto h-[5px] w-[146px] rounded-full bg-[#D7DEE8]" />;
}

function MobileTopBar({ title }: { title: string }) {
  return (
    <div className="flex h-[56px] items-center px-4">
      <div className="text-[18px] font-bold tracking-[-0.18px] text-[#F3F6FB]">{title}</div>
    </div>
  );
}

function ReviewScreen({
  data,
  onSubmit,
  locale,
  isLoading = false,
}: {
  data: SandboxFormState;
  onSubmit: () => void;
  locale: Locale;
  isLoading?: boolean;
}) {
  const amountLabel = formatSandboxAmount(data.amount, data.currency);
  const walletLabel = inferWalletDisplayName(data.walletId);
  const recipientName = inferRecipientName(data.merchantName);
  const txTypeLabel =
    data.txType === 'P2P'
      ? demoText(locale, 'demo.tx.peer')
      : data.txType === 'CASH_IN'
        ? demoText(locale, 'demo.tx.cashIn')
        : data.txType === 'CASH_OUT'
          ? demoText(locale, 'demo.tx.cashOut')
          : demoText(locale, 'demo.tx.merchant');

  return (
    <div className="flex h-full flex-col bg-[#10141A]">
      <StatusBar />
      <MobileTopBar title={demoText(locale, 'demo.phone.review')} />

      <div className="flex flex-1 flex-col px-4 pt-[18px] text-[#DFE2EB]">
        <div>
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-[14px]">
              <div className="flex h-[42px] w-[42px] items-center justify-center rounded-[12px] bg-[#222833]">
                <Wallet size={20} color="#ADC6FF" strokeWidth={2} aria-hidden="true" />
              </div>
              <div>
                <div className="text-[12px] font-medium tracking-[0.02em] text-[#8C909F]">{demoText(locale, 'demo.phone.from')}</div>
                <div className="mt-[2px] text-[16px] font-semibold tracking-[-0.16px] text-[#DFE2EB]">{walletLabel}</div>
              </div>
            </div>
            <button
              type="button"
              className="rounded-[10px] border border-[#4D8EFF]/20 bg-[#4D8EFF]/10 px-3 py-[7px] text-[12px] font-semibold tracking-[-0.12px] text-[#ADC6FF] shadow-none"
              style={{ minHeight: 0 }}
            >
              {demoText(locale, 'demo.phone.change')}
            </button>
          </div>
        </div>

        <div className="flex items-center gap-5 px-[10px] py-4">
          <div className="flex items-center justify-center text-[#4D8EFF]">
            <ArrowDown size={16} strokeWidth={1.8} />
          </div>
          <div className="h-px flex-1 bg-white/[0.06]" />
        </div>

        <div className="rounded-[18px] border border-white/[0.06] bg-[#1C2026] px-4 py-4">
          <div className="flex items-center gap-[14px]">
            <RecipientAvatar />
            <div>
              <div className="text-[16px] font-semibold tracking-[-0.16px] text-[#DFE2EB]">{recipientName}</div>
              <div className="mt-[2px] flex items-center gap-[6px] text-[12px] font-normal tracking-[-0.12px] text-[#8C909F]">
                <span>{txTypeLabel}</span>
                <span className="h-1 w-1 rounded-full bg-[#C4C4C4]" />
                <span>{data.userId || 'user_1001'}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 rounded-[18px] border border-white/[0.06] bg-[#1C2026] px-5 py-5">
          <div className="text-[12px] font-medium uppercase tracking-[0.14em] text-[#8C909F]">{demoText(locale, 'demo.phone.summary')}</div>
          <div className="mt-5 space-y-3">
            <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
              <span className="text-[#8C909F]">{sharedText(locale, 'form.field.paymentType')}</span>
              <span className="font-medium text-[#DFE2EB]">{txTypeLabel}</span>
            </div>
            <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
              <span className="text-[#8C909F]">{demoText(locale, 'demo.field.channel')}</span>
              <span className="font-medium text-[#DFE2EB]">{data.channel}</span>
            </div>
            <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
              <span className="text-[#8C909F]">{demoText(locale, 'demo.phone.corridor')}</span>
              <span className="font-medium text-[#DFE2EB]">
                {data.isCrossBorder
                  ? `${data.sourceCountry} → ${data.destinationCountry}`
                  : data.sourceCountry}
              </span>
            </div>
            <div className="flex items-center justify-between gap-4 text-[14px] tracking-[-0.14px]">
              <span className="text-[#8C909F]">{sharedText(locale, 'form.field.currency')}</span>
              <span className="font-medium text-[#DFE2EB]">{data.currency || 'USD'}</span>
            </div>
          </div>
          <div className="my-4 h-px bg-white/[0.06]" />
          <div className="flex items-center justify-between gap-4">
            <span className="text-[14px] font-normal tracking-[-0.14px] text-[#8C909F]">{demoText(locale, 'demo.phone.total')}</span>
            <span className="text-[28px] font-black tracking-[-0.28px] text-[#DFE2EB]">{amountLabel}</span>
          </div>
        </div>
      </div>

      <div className="border-t border-white/[0.06] bg-[#10141A] px-4 pb-2 pt-5">
        <button
          type="button"
          onClick={onSubmit}
          disabled={isLoading}
          className="h-[48px] w-full rounded-[12px] border-0 bg-[#ADC6FF] text-[16px] font-bold tracking-[-0.16px] text-[#002E6A] shadow-[0px_10px_15px_-3px_rgba(173,198,255,0.2),0px_4px_6px_-4px_rgba(173,198,255,0.2)] transition-transform duration-200 hover:scale-[0.99] disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <span style={{ display: 'inline-block', animation: 'integration-spin 0.7s linear infinite' }} aria-label={demoText(locale, 'demo.phone.processing')}>○</span>
          ) : demoText(locale, 'demo.phone.pay')}
        </button>
        <div className="pb-[8.75px] pt-5">
          <HomeIndicator />
        </div>
      </div>
    </div>
  );
}

function AnimatedScreen({ screenKey, children }: { screenKey: string; children: React.ReactNode }) {
  return (
    <div key={screenKey} className="h-full animate-[integration-slide-in_420ms_cubic-bezier(0.22,1,0.36,1)]">
      {children}
    </div>
  );
}

function parseNumericValue(value: string): number {
  const parsed = Number.parseFloat(value.replace(/[^0-9.]/g, ''));
  return Number.isFinite(parsed) ? parsed : 0;
}

function deriveDecisionFromForm(form: SandboxFormState): Decision {
  const composite = deriveRiskScoreFromForm(form);

  if (composite >= 0.78) return 'BLOCK';
  if (composite >= 0.42) return 'FLAG';
  return 'APPROVE';
}

function deriveRiskScoreFromForm(form: SandboxFormState): number {
  const avgRisk =
    (parseNumericValue(form.deviceRiskScore) +
      parseNumericValue(form.ipRiskScore) +
      parseNumericValue(form.locationRiskScore)) /
    3;
  const sharedUsers = parseNumericValue(form.deviceSharedUsers24h);
  const accountAge = parseNumericValue(form.accountAgeDays);
  const cashVelocity = parseNumericValue(form.cashFlowVelocity1h);
  const counterparties = parseNumericValue(form.p2pCounterparties24h);
  const amount = parseNumericValue(form.amount);

  let composite = avgRisk;
  composite += form.simChangeRecent ? 0.14 : 0;
  composite += form.isCrossBorder ? 0.12 : 0;
  composite += sharedUsers >= 3 ? Math.min((sharedUsers - 2) * 0.035, 0.16) : 0;
  composite += accountAge < 30 ? 0.12 : accountAge < 90 ? 0.05 : 0;
  composite += cashVelocity >= 4 ? Math.min((cashVelocity - 3) * 0.028, 0.16) : 0;
  composite += counterparties >= 3 ? Math.min((counterparties - 2) * 0.018, 0.12) : 0;
  composite += amount >= 750 ? 0.08 : amount >= 250 ? 0.04 : 0;
  composite += form.isAgentAssisted ? 0.06 : 0;
  composite += form.connectivityMode === 'offline_buffered' ? 0.04
             : form.connectivityMode === 'intermittent'     ? 0.02
             : 0;

  return Math.min(0.99, Math.max(0.01, Number(composite.toFixed(3))));
}

function deriveReasonsFromDecision(decision: Decision, form: SandboxFormState): string[] {
  const reasons: string[] = [];

  if (decision !== 'APPROVE') {
    reasons.push('High fraud probability from transaction model');
  }
  if (parseNumericValue(form.amount) >= 250) {
    reasons.push('Above-normal transaction amount');
  }
  if (form.isCrossBorder || form.simChangeRecent || parseNumericValue(form.cashFlowVelocity1h) >= 4) {
    reasons.push('Context contributed to elevated risk');
  }
  if (decision === 'APPROVE' || reasons.length === 0) {
    reasons.push("Behavior is consistent with the user's normal baseline");
  }

  return reasons;
}

function formatSandboxAmount(amount: string, currency: string): string {
  const numeric = parseNumericValue(amount);
  if (!numeric) return currency ? `${currency} 0.00` : '$0.00';
  if (currency.toUpperCase() === 'IDR') return formatAmountForPreview(amount);
  return `${currency || 'USD'} ${numeric.toFixed(2)}`;
}

export default function IntegrationSandbox() {
  const [locale, setLocale] = useState<Locale>('en');
  const [form, setForm] = useState<SandboxFormState>(createInitialFormState);
  const [mobileStage, setMobileStage] = useState<'review' | 'result'>('review');
  const [apiLoading, setApiLoading] = useState(false);
  const [apiResult, setApiResult] = useState<ScoreTransactionResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const resetApiState = () => { setApiResult(null); setApiError(null); };

  const updateField = <K extends keyof SandboxFormState>(key: K, value: SandboxFormState[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
    setMobileStage('review');
    resetApiState();
  };

  const updateSourceCountry = (country: CountryCode) => {
    setForm((current) => ({
      ...current,
      sourceCountry: country,
      currency: COUNTRY_CURRENCY[country],
    }));
    setMobileStage('review');
    resetApiState();
  };

  const liveDecision = useMemo(() => deriveDecisionFromForm(form), [form]);
  const liveRiskScore = useMemo(() => deriveRiskScoreFromForm(form), [form]);
  const liveTransaction = useMemo<Transaction>(
    () => ({
      merchant: inferRecipientName(form.merchantName),
      amount: formatSandboxAmount(form.amount, form.currency),
      type: form.txType,
    }),
    [form.amount, form.currency, form.merchantName, form.txType],
  );
  const liveReasons = useMemo(() => deriveReasonsFromDecision(liveDecision, form), [form, liveDecision]);

  const displayDecision: Decision = (apiResult?.decision ?? liveDecision) as Decision;
  const displayReasons = apiResult?.reasons ?? apiResult?.fraud_reasons ?? liveReasons;
  const translate = (key: string): string => sharedText(locale, key);
  const demo = (key: string): string => demoText(locale, key);

  const handleSubmitReview = async () => {
    setApiLoading(true);
    setApiError(null);
    setApiResult(null);
    setMobileStage('result');

    let result: ScoreTransactionResponse | null = null;
    try {
      const payload: ScoreTransactionPayload = {
        schema_version: 'ieee_fraud_tx_v1',
        user_id: form.userId || 'sandbox_user',
        transaction_amount: parseNumericValue(form.amount),
        currency: form.currency || undefined,
        device_risk_score: parseNumericValue(form.deviceRiskScore),
        ip_risk_score: parseNumericValue(form.ipRiskScore),
        location_risk_score: parseNumericValue(form.locationRiskScore),
        device_id: form.deviceId || 'sandbox_device',
        device_shared_users_24h: Math.round(parseNumericValue(form.deviceSharedUsers24h)),
        account_age_days: Math.round(parseNumericValue(form.accountAgeDays)),
        sim_change_recent: form.simChangeRecent,
        tx_type: form.txType,
        channel: form.channel,
        cash_flow_velocity_1h: Math.round(parseNumericValue(form.cashFlowVelocity1h)),
        p2p_counterparties_24h: Math.round(parseNumericValue(form.p2pCounterparties24h)),
        is_cross_border: form.isCrossBorder,
        source_country: form.sourceCountry,
        destination_country: form.isCrossBorder ? form.destinationCountry : form.sourceCountry,
        is_agent_assisted: form.isAgentAssisted,
        connectivity_mode: form.connectivityMode,
      };
      result = await scoreTransaction(payload);
      setApiResult(result);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : demoText(locale, 'demo.status.apiUnavailable'));
    } finally {
      setApiLoading(false);
    }

    const decision = (result?.decision ?? liveDecision) as Decision;
    const riskScore = result?.final_risk_score ?? liveRiskScore;
    addTransactionToHistory({
      id: `tx_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`,
      userId: form.userId || 'Anonymous',
      status: decision,
      transactionType: form.txType,
      amount: parseNumericValue(form.amount).toFixed(2),
      merchantName: inferRecipientName(form.merchantName),
      walletId: inferWalletDisplayName(form.walletId),
      currency: form.currency || 'USD',
      crossBorder: form.isCrossBorder,
      timestamp: new Date().toISOString(),
      riskScore,
      explainabilityBase: result?.explainability?.base ?? parseNumericValue(form.deviceRiskScore),
      explainabilityContext: result?.explainability?.context ?? parseNumericValue(form.ipRiskScore),
      explainabilityBehavior: result?.explainability?.behavior ?? parseNumericValue(form.locationRiskScore),
    });
  };

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#0C0E12] text-[#F7F7F7]">
      <style>
        {`
          @keyframes integration-slide-in {
            0% { opacity: 0; transform: translateX(28px); }
            100% { opacity: 1; transform: translateX(0); }
          }

          @keyframes integration-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>

      <div className="grid w-full gap-6 xl:grid-cols-[minmax(380px,480px)_minmax(0,1fr)]">
        <section className="rounded-[28px] border border-white/10 bg-white/[0.06] py-6 shadow-[0_24px_80px_rgba(0,0,0,0.32)] backdrop-blur-xl xl:sticky xl:top-0 xl:flex xl:h-screen xl:flex-col">
          <div className="px-6">
            <div className="flex items-center justify-between gap-3">
              <p className="text-[12px] font-semibold uppercase tracking-[0.18em] text-[#94979C]">{demo('demo.eyebrow')}</p>
              <div className="relative flex shrink-0 items-center rounded-full border border-white/8 bg-white/[0.04] pl-2 pr-7 text-[#94979C]">
                <Globe size={13} aria-hidden="true" />
                <label htmlFor="demo-language" className="sr-only">
                  {demo('demo.language')}
                </label>
                <select
                  id="demo-language"
                  value={locale}
                  onChange={(event) => setLocale(event.target.value as Locale)}
                  className="h-8 max-w-[86px] appearance-none bg-transparent px-1 text-[11px] font-medium text-[#CECFD2] outline-none"
                  title={demo('demo.language')}
                >
                  {localeOptions.map((option) => (
                    <option key={option.code} value={option.code}>
                      {translate(option.labelKey)}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[#94979C]">
                  <ArrowDown size={12} />
                </div>
              </div>
            </div>
            <h1 className="mt-3 max-w-[360px] text-[32px] font-semibold leading-[1.05] tracking-[-0.03em] text-white">
              {demo('demo.title')}
            </h1>
            <p className="mt-4 max-w-[360px] text-[15px] leading-7 text-[#CECFD2]">
              {demo('demo.subtitle')}
            </p>
          </div>

          <div className="mt-8 overflow-y-auto px-6 pb-8 xl:mt-6 xl:min-h-0 xl:flex-1">
            <div className="rounded-[20px] border border-white/8 bg-[#11151C] p-5">
              <div className="mb-4 text-[18px] font-semibold text-white">{translate('form.title')}</div>
              <p className="mb-4 text-[13px] leading-6 text-[#94979C]">
                {demo('demo.subtitle')}
              </p>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {formScenarioMeta.map((scenario) => {
                  const active = form.formScenario === scenario.id;
                  return (
                    <button
                      key={scenario.id}
                      type="button"
                      onClick={() => {
                        setForm((current) => applyScenarioPreset(current, scenario.id));
                        setMobileStage('review');
                        resetApiState();
                      }}
                      className="rounded-[16px] border p-4 text-left transition"
                      style={{
                        borderColor: active ? '#1273E7' : 'rgba(255,255,255,0.08)',
                        background: active ? 'rgba(18,115,231,0.16)' : 'rgba(255,255,255,0.03)',
                      }}
                    >
                      <scenario.Icon size={18} className={active ? 'text-[#BFD8FF]' : 'text-[#CECFD2]'} />
                      <div className={`mt-4 text-[15px] font-semibold ${active ? 'text-[#BFD8FF]' : 'text-white'}`}>{translate(scenario.labelKey)}</div>
                      <div className="mt-1 text-[13px] leading-5 text-[#94979C]">{translate(scenario.descriptionKey)}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-4 rounded-[20px] border border-white/8 bg-[#11151C] p-5">
              <div className="mb-4 text-[18px] font-semibold text-white">{translate('form.section.paymentDetails')}</div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <FieldLabel>{translate('form.field.userId')}</FieldLabel>
                  <FieldInput value={form.userId} onChange={(value) => updateField('userId', value)} placeholder={translate('form.placeholder.userId')} />
                </div>
                <div>
                  <FieldLabel>{translate('form.field.amount')}</FieldLabel>
                  <FieldInput value={form.amount} onChange={(value) => updateField('amount', value)} placeholder="250000" />
                </div>
              </div>
              <div className="mt-4">
                <FieldLabel>{translate('form.field.paymentType')}</FieldLabel>
                <FieldSelect<TxType>
                  value={form.txType}
                  onChange={(value) => updateField('txType', value)}
                  options={[
                    { value: 'MERCHANT', label: demo('demo.tx.merchant') },
                    { value: 'P2P', label: demo('demo.tx.peer') },
                    { value: 'CASH_IN', label: demo('demo.tx.cashIn') },
                    { value: 'CASH_OUT', label: demo('demo.tx.cashOut') },
                  ]}
                />
              </div>
              <button
                type="button"
                onClick={() => updateField('isCrossBorder', !form.isCrossBorder)}
                className="mt-4 flex items-center gap-3 border-0 bg-transparent p-0 text-left shadow-none"
                style={{ minHeight: 0 }}
              >
                <div className={`flex h-[18px] w-[18px] items-center justify-center rounded-[4px] border ${form.isCrossBorder ? 'border-[#1273E7] bg-[#1273E7]' : 'border-[#6B7280]'}`}>
                  {form.isCrossBorder ? <CheckCircle2 size={12} color="white" /> : null}
                </div>
                <span className="text-[13px] text-white">{translate('form.field.crossBorder')}</span>
              </button>

              {/* ASEAN context — source/destination countries */}
              <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <FieldLabel>{demo('demo.field.sourceCountry')}</FieldLabel>
                  <FieldSelect<CountryCode>
                    value={form.sourceCountry}
                    onChange={updateSourceCountry}
                    options={[
                      { value: 'SG', label: 'SG — Singapore' },
                      { value: 'MY', label: 'MY — Malaysia' },
                      { value: 'ID', label: 'ID — Indonesia' },
                      { value: 'TH', label: 'TH — Thailand' },
                      { value: 'PH', label: 'PH — Philippines' },
                      { value: 'VN', label: 'VN — Vietnam' },
                    ]}
                  />
                </div>
                <div style={{ opacity: form.isCrossBorder ? 1 : 0.5 }}>
                  <FieldLabel>{demo('demo.field.destinationCountry')}</FieldLabel>
                  <FieldSelect<CountryCode>
                    value={form.isCrossBorder ? form.destinationCountry : form.sourceCountry}
                    onChange={(value) => updateField('destinationCountry', value)}
                    options={[
                      { value: 'SG', label: 'SG — Singapore' },
                      { value: 'MY', label: 'MY — Malaysia' },
                      { value: 'ID', label: 'ID — Indonesia' },
                      { value: 'TH', label: 'TH — Thailand' },
                      { value: 'PH', label: 'PH — Philippines' },
                      { value: 'VN', label: 'VN — Vietnam' },
                    ]}
                  />
                </div>
              </div>

              {/* ASEAN context — connectivity + agent */}
              <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <FieldLabel>{demo('demo.field.connectivity')}</FieldLabel>
                  <FieldSelect<ConnectivityMode>
                    value={form.connectivityMode}
                    onChange={(value) => updateField('connectivityMode', value)}
                    options={[
                      { value: 'online', label: demo('demo.connectivity.live') },
                      { value: 'intermittent', label: demo('demo.connectivity.intermittent') },
                      { value: 'offline_buffered', label: demo('demo.connectivity.degraded') },
                    ]}
                  />
                </div>
                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={() => updateField('isAgentAssisted', !form.isAgentAssisted)}
                    className="flex items-center gap-3 border-0 bg-transparent p-0 text-left shadow-none"
                    style={{ minHeight: 0 }}
                  >
                    <div className={`flex h-[18px] w-[18px] items-center justify-center rounded-[4px] border ${form.isAgentAssisted ? 'border-[#1273E7] bg-[#1273E7]' : 'border-[#6B7280]'}`}>
                      {form.isAgentAssisted ? <CheckCircle2 size={12} color="white" /> : null}
                    </div>
                    <span className="text-[13px] text-white">{demo('demo.field.agentAssisted')}</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="mt-4 rounded-[20px] border border-white/8 bg-[#11151C] p-5">
              <div className="mb-2 text-[18px] font-semibold text-white">{demo('demo.section.advanced')}</div>
              <p className="mb-4 text-[13px] leading-6 text-[#94979C]">
                {demo('demo.section.advancedSubtitle')}
              </p>

              <div className="mb-3 text-[12px] font-medium uppercase tracking-[0.14em] text-[#94979C]">{demo('demo.section.riskScores')}</div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <FieldLabel>{demo('demo.field.deviceRisk')}</FieldLabel>
                  <FieldInput value={form.deviceRiskScore} onChange={(value) => updateField('deviceRiskScore', value)} />
                </div>
                <div>
                  <FieldLabel>{demo('demo.field.ipRisk')}</FieldLabel>
                  <FieldInput value={form.ipRiskScore} onChange={(value) => updateField('ipRiskScore', value)} />
                </div>
                <div>
                  <FieldLabel>{demo('demo.field.locationRisk')}</FieldLabel>
                  <FieldInput value={form.locationRiskScore} onChange={(value) => updateField('locationRiskScore', value)} />
                </div>
              </div>

              <div className="mb-3 mt-6 text-[12px] font-medium uppercase tracking-[0.14em] text-[#94979C]">{demo('demo.section.deviceSignals')}</div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <FieldLabel>{demo('demo.field.deviceId')}</FieldLabel>
                  <FieldInput value={form.deviceId} onChange={(value) => updateField('deviceId', value)} />
                </div>
                <div>
                  <FieldLabel>{demo('demo.field.deviceShared')}</FieldLabel>
                  <FieldInput value={form.deviceSharedUsers24h} onChange={(value) => updateField('deviceSharedUsers24h', value)} />
                </div>
                <div>
                  <FieldLabel>{demo('demo.field.accountAge')}</FieldLabel>
                  <FieldInput value={form.accountAgeDays} onChange={(value) => updateField('accountAgeDays', value)} />
                </div>
                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={() => updateField('simChangeRecent', !form.simChangeRecent)}
                    className="flex items-center gap-3 border-0 bg-transparent p-0 text-left shadow-none"
                    style={{ minHeight: 0 }}
                  >
                    <div className={`flex h-[18px] w-[18px] items-center justify-center rounded-[4px] border ${form.simChangeRecent ? 'border-[#1273E7] bg-[#1273E7]' : 'border-[#6B7280]'}`}>
                      {form.simChangeRecent ? <CheckCircle2 size={12} color="white" /> : null}
                    </div>
                    <span className="text-[13px] text-white">{demo('demo.field.simChanged')}</span>
                  </button>
                </div>
              </div>

              <div className="mb-3 mt-6 text-[12px] font-medium uppercase tracking-[0.14em] text-[#94979C]">{demo('demo.section.behavior')}</div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <FieldLabel>{demo('demo.field.channel')}</FieldLabel>
                  <FieldSelect<Channel>
                    value={form.channel}
                    onChange={(value) => updateField('channel', value)}
                    options={[
                      { value: 'APP', label: 'APP' },
                      { value: 'WEB', label: 'WEB' },
                      { value: 'AGENT', label: 'AGENT' },
                      { value: 'QR', label: 'QR' },
                    ]}
                  />
                </div>
                <div>
                  <FieldLabel>{demo('demo.field.cashVelocity')}</FieldLabel>
                  <FieldInput value={form.cashFlowVelocity1h} onChange={(value) => updateField('cashFlowVelocity1h', value)} />
                </div>
                <div>
                  <FieldLabel>{demo('demo.field.counterparties')}</FieldLabel>
                  <FieldInput value={form.p2pCounterparties24h} onChange={(value) => updateField('p2pCounterparties24h', value)} />
                </div>
              </div>
            </div>

            <div className="mt-4 rounded-[20px] border border-white/8 bg-[#11151C] p-5">
              <div className="mb-4 text-[18px] font-semibold text-white">{translate('form.section.wallet')}</div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <FieldLabel>{translate('form.field.walletId')}</FieldLabel>
                  <FieldInput value={form.walletId} onChange={(value) => updateField('walletId', value)} placeholder={translate('form.placeholder.walletId')} />
                </div>
                <div>
                  <FieldLabel>{translate('form.field.currency')}</FieldLabel>
                  <FieldSelect<string>
                    value={form.currency}
                    onChange={(value) => updateField('currency', value)}
                    options={ASEAN_CURRENCIES.map((c) => ({ value: c, label: c }))}
                  />
                </div>
              </div>
              <div className="mt-4">
                <FieldLabel>{translate('form.field.merchant')}</FieldLabel>
                <FieldInput value={form.merchantName} onChange={(value) => updateField('merchantName', value)} placeholder={translate('form.placeholder.merchant')} />
              </div>
            </div>
          </div>
        </section>

        <section className="xl:sticky xl:top-8 xl:self-start flex min-h-[780px] items-start justify-center rounded-[32px] border border-white/6 bg-[radial-gradient(circle_at_top,rgba(120,76,252,0.08),transparent_34%),linear-gradient(180deg,#0F1218_0%,#0C0E12_100%)] px-4 py-8 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] lg:px-6">
          <div className="flex w-full max-w-[620px] flex-col items-center gap-6">
            {mobileStage === 'result' && (apiResult || apiError || apiLoading) && (
              <div className="w-full max-w-[430px] rounded-[20px] border border-white/8 bg-[#11151C] p-5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#94979C]">
                    {apiLoading ? demo('demo.status.calling') : apiResult ? demo('demo.status.liveResponse') : demo('demo.status.offlineEstimate')}
                  </span>
                  {apiResult?.runtime_mode && (
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                      apiResult.runtime_mode === 'primary' ? 'bg-emerald-900/50 text-emerald-400' :
                      apiResult.runtime_mode === 'cached_context' ? 'bg-yellow-900/50 text-yellow-400' :
                      'bg-red-900/50 text-red-400'
                    }`}>
                      {apiResult.runtime_mode === 'primary' ? demo('demo.status.live') :
                       apiResult.runtime_mode === 'cached_context' ? demo('demo.status.cached') : demo('demo.status.degraded')}
                    </span>
                  )}
                </div>
                {apiError && (
                  <p className="mt-2 text-[12px] text-[#fca5a5]">{apiError} - {demo('demo.status.showingEstimate')}</p>
                )}
                {apiResult && (
                  <div className="mt-3 space-y-2">
                    {apiResult.decision_source && (
                      <div className="flex justify-between text-[13px]">
                        <span className="text-[#8C909F]">{demo('demo.status.decisionSource')}</span>
                        <span className="font-medium text-[#DFE2EB]">{apiResult.decision_source.replace(/_/g, ' ')}</span>
                      </div>
                    )}
                    {apiResult.corridor && (
                      <div className="flex justify-between text-[13px]">
                        <span className="text-[#8C909F]">{demo('demo.phone.corridor')}</span>
                        <span className="font-medium text-[#DFE2EB]">{apiResult.corridor}</span>
                      </div>
                    )}
                    {apiResult.normalized_amount_reference != null && (
                      <div className="flex justify-between text-[13px]">
                        <span className="text-[#8C909F]">{demo('demo.status.normalisedAmount')}</span>
                        <span className="font-medium text-[#DFE2EB]">
                          {apiResult.normalized_amount_reference.toFixed(2)}
                          {apiResult.normalization_basis ? (
                            <span className="ml-1 text-[11px] text-[#8C909F]">({apiResult.normalization_basis})</span>
                          ) : null}
                        </span>
                      </div>
                    )}
                    {apiResult.explainability?.ring != null && apiResult.explainability.ring > 0 && (
                      <div className="flex justify-between text-[13px]">
                        <span className="text-[#8C909F]">{demo('demo.status.ringScore')}</span>
                        <span className="font-medium text-orange-400">{apiResult.explainability.ring.toFixed(3)}</span>
                      </div>
                    )}
                    {apiResult.ring_match_type && apiResult.ring_match_type !== 'none' && (
                      <div className="flex justify-between text-[13px]">
                        <span className="text-[#8C909F]">{demo('demo.status.ringMatch')}</span>
                        <span className="font-medium text-orange-400">
                          {apiResult.ring_match_type === 'account_member' ? demo('demo.status.accountRing') : demo('demo.status.attributeMatch')}
                        </span>
                      </div>
                    )}
                    {apiResult.stage_timings_ms?.total_pipeline_ms != null && (
                      <div className="flex justify-between text-[13px]">
                        <span className="text-[#8C909F]">{demo('demo.status.latency')}</span>
                        <span className="font-medium text-[#DFE2EB]">{Math.round(apiResult.stage_timings_ms.total_pipeline_ms)} ms</span>
                      </div>
                    )}
                    {apiResult.explainability?.top_feature_drivers && apiResult.explainability.top_feature_drivers.length > 0 && (
                      <div className="pt-1">
                        <p className="mb-1.5 text-[11px] text-[#8C909F]">{demo('demo.status.topDrivers')}</p>
                        <div className="space-y-1">
                          {apiResult.explainability.top_feature_drivers.slice(0, 4).map((d) => (
                            <div key={d.feature} className="flex items-center justify-between text-[12px]">
                              <span className="font-mono text-[#CECFD2]">{d.feature}</span>
                              <span className={d.direction === 'increases_risk' ? 'text-red-400' : 'text-emerald-400'}>
                                {d.direction === 'increases_risk' ? '+' : '−'}{Math.abs(d.shap_value).toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {apiResult.reason_codes && apiResult.reason_codes.length > 0 && (
                      <div className="pt-1">
                        <p className="mb-1.5 text-[11px] text-[#8C909F]">{demo('demo.status.reasonCodes')}</p>
                        <div className="flex flex-wrap gap-1.5">
                          {apiResult.reason_codes.map((code) => (
                            <span key={code} className="rounded-md bg-white/5 px-2 py-0.5 text-[10px] font-mono text-[#CECFD2]">
                              {code}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            <div className="h-[820px] w-[430px] max-w-full overflow-hidden rounded-[32px] border border-white/10 bg-[#10141A] shadow-[0_35px_100px_rgba(0,0,0,0.45)]">
              <AnimatedScreen screenKey={mobileStage}>
                {mobileStage === 'review' ? (
                  <ReviewScreen data={form} onSubmit={handleSubmitReview} locale={locale} isLoading={apiLoading} />
                ) : (
                  <FraudResultScreen
                    embedded
                    locale={locale}
                    decision={apiLoading ? liveDecision : displayDecision}
                    transaction={liveTransaction}
                    reasons={apiLoading ? liveReasons : displayReasons}
                    onGoBack={() => { setMobileStage('review'); resetApiState(); }}
                    onCheckAnother={() => { setMobileStage('review'); resetApiState(); }}
                  />
                )}
              </AnimatedScreen>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
