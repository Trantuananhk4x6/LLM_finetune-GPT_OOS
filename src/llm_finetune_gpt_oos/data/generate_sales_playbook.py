import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import load_config


SYSTEM_PROMPT = "Bạn là tư vấn viên bán hàng chuyên nghiệp cho ngành nội thất và điện máy. Tư vấn dựa trên nhu cầu thực, nói rõ điều chưa biết, xác nhận giá và chính sách trước khi cam kết, tôn trọng quyền quyết định của khách hàng. Không dùng khan hiếm giả, không che giấu chi phí, không gây áp lực và không khai thác điểm yếu cá nhân."
PRODUCTS = [
    {"vertical": "dien_may", "product": "tivi", "focus": "kích thước màn hình, khoảng cách xem, độ phân giải và vị trí treo", "proof": "mã model, kích thước thực, cổng kết nối và điều kiện lắp đặt"},
    {"vertical": "dien_may", "product": "tủ lạnh", "focus": "dung tích, cách chia ngăn, kích thước lọt lòng và khoảng thoáng tản nhiệt", "proof": "dung tích sử dụng, kích thước thực tế và điều kiện bảo hành"},
    {"vertical": "dien_may", "product": "máy giặt", "focus": "khối lượng giặt, loại vải, vị trí cấp thoát nước và không gian mở cửa", "proof": "khối lượng giặt, kích thước lắp đặt và chương trình giặt"},
    {"vertical": "dien_may", "product": "máy sấy", "focus": "khối lượng sấy, loại vải, thoát khí hoặc vị trí đặt máy", "proof": "công nghệ sấy, kích thước và điều kiện lắp đặt"},
    {"vertical": "dien_may", "product": "điều hòa", "focus": "diện tích, trần cao, hướng nắng, số người và vị trí dàn nóng", "proof": "công suất phù hợp, vật tư lắp đặt và khảo sát kỹ thuật"},
    {"vertical": "dien_may", "product": "máy lọc nước", "focus": "nguồn nước đầu vào, nhu cầu uống trực tiếp, vị trí lắp và lịch thay lõi", "proof": "công nghệ lọc, chi phí lõi thay thế và điều kiện bảo hành"},
    {"vertical": "dien_may", "product": "nồi chiên không dầu", "focus": "số người dùng, dung tích, không gian bếp và thói quen nấu", "proof": "dung tích thực, phụ kiện đi kèm và hướng dẫn vệ sinh"},
    {"vertical": "dien_may", "product": "bếp từ", "focus": "kích thước khoét đá, loại nồi hiện có, công suất điện và nhu cầu nấu", "proof": "kích thước khoét đá, công suất và phạm vi lắp đặt"},
    {"vertical": "dien_may", "product": "robot hút bụi", "focus": "diện tích sàn, thảm, thú cưng, ngưỡng cửa và nhu cầu tự động", "proof": "khả năng vượt ngưỡng, chế độ làm sạch và phụ kiện tiêu hao"},
    {"vertical": "dien_may", "product": "máy hút ẩm", "focus": "diện tích phòng, độ ẩm, vị trí đặt máy và nhu cầu sấy quần áo", "proof": "công suất hút ẩm, độ ồn và dung tích bình nước"},
    {"vertical": "dien_may", "product": "máy lọc không khí", "focus": "diện tích phòng, tình trạng bụi mịn, mùi, dị ứng và chi phí màng lọc", "proof": "diện tích khuyến nghị, loại màng lọc và lịch thay thế"},
    {"vertical": "dien_may", "product": "laptop", "focus": "phần mềm cần dùng, tính di động, thời lượng pin, màn hình và ngân sách", "proof": "cấu hình, cổng kết nối, bảo hành và khả năng nâng cấp"},
    {"vertical": "dien_may", "product": "màn hình", "focus": "công việc, kích thước bàn, độ phân giải, tần số quét và cổng kết nối", "proof": "thông số hiển thị, khả năng tương thích và chính sách điểm chết"},
    {"vertical": "dien_may", "product": "máy pha cà phê", "focus": "số ly mỗi ngày, loại hạt, thao tác vệ sinh và không gian quầy", "proof": "công suất, quy trình vệ sinh và phụ kiện thay thế"},
    {"vertical": "dien_may", "product": "camera giám sát", "focus": "vị trí cần quan sát, góc nhìn, lưu trữ, mạng internet và quyền riêng tư", "proof": "góc nhìn, chuẩn lưu trữ, phí dịch vụ và điều kiện lắp đặt"},
    {"vertical": "dien_may", "product": "máy nước nóng", "focus": "số người dùng, công suất điện, áp lực nước và vị trí lắp", "proof": "tính năng an toàn, công suất, khảo sát dây điện và bảo hành"},
    {"vertical": "noi_that", "product": "sofa", "focus": "kích thước phòng, lối đi, số người ngồi, vật liệu bọc và thói quen sử dụng", "proof": "kích thước phủ bì, chất liệu khung, mút và điều kiện giao lắp"},
    {"vertical": "noi_that", "product": "bàn ăn", "focus": "số người dùng, diện tích lối đi, kiểu ghế và vật liệu mặt bàn", "proof": "kích thước mở rộng, vật liệu, tải trọng và hướng dẫn bảo quản"},
    {"vertical": "noi_that", "product": "giường ngủ", "focus": "diện tích phòng, kích thước nệm, chiều cao mong muốn và nhu cầu chứa đồ", "proof": "kích thước lòng giường, kết cấu và điều kiện lắp ráp"},
    {"vertical": "noi_that", "product": "tủ quần áo", "focus": "kích thước tường, kiểu mở cánh, số lượng đồ và vị trí ổ điện", "proof": "kích thước, ray trượt, vật liệu và phương án lắp đặt"},
    {"vertical": "noi_that", "product": "bàn làm việc", "focus": "thiết bị đặt trên bàn, chiều cao ghế, quản lý dây điện và diện tích", "proof": "kích thước mặt bàn, tải trọng và không gian đi dây"},
    {"vertical": "noi_that", "product": "ghế công thái học", "focus": "chiều cao người dùng, thời gian ngồi, điểm tựa lưng và kích thước bàn", "proof": "dải điều chỉnh, tải trọng, vật liệu lưới và bảo hành cơ cấu"},
    {"vertical": "noi_that", "product": "kệ tivi", "focus": "kích thước tivi, vị trí dây điện, tải trọng và phong cách phòng", "proof": "kích thước, tải trọng, lỗ đi dây và phương án lắp"},
    {"vertical": "noi_that", "product": "tủ giày", "focus": "số đôi giày, độ sâu lối vào, nhu cầu thông thoáng và kích thước cửa", "proof": "sức chứa, chiều sâu, vật liệu và kích thước thực tế"},
    {"vertical": "noi_that", "product": "bàn trà", "focus": "chiều cao sofa, lối đi, nhu cầu chứa đồ và an toàn trẻ nhỏ", "proof": "kích thước, bo góc, vật liệu và hướng dẫn bảo quản"},
    {"vertical": "noi_that", "product": "nệm", "focus": "tư thế ngủ, độ cứng mong muốn, kích thước giường và tình trạng sức khỏe", "proof": "độ cứng, kích thước, vật liệu, điều kiện đổi thử và bảo hành"},
    {"vertical": "noi_that", "product": "rèm cửa", "focus": "kích thước cửa, hướng nắng, mức cản sáng, kiểu treo và vệ sinh", "proof": "khổ vải, mức cản sáng, thời gian may đo và lắp đặt"},
    {"vertical": "noi_that", "product": "tủ bếp", "focus": "mặt bằng bếp, tam giác công năng, thiết bị âm tủ và ngân sách", "proof": "bản vẽ, vật liệu, phụ kiện, tiến độ và điều khoản nghiệm thu"},
    {"vertical": "noi_that", "product": "kệ sách", "focus": "số lượng sách, chiều cao trần, tải trọng và vị trí neo tường", "proof": "tải trọng từng tầng, kích thước, neo tường và vật liệu"},
    {"vertical": "noi_that", "product": "đèn trang trí", "focus": "diện tích, mục đích chiếu sáng, màu ánh sáng, trần nhà và công tắc", "proof": "công suất, kích thước, loại đui đèn và điều kiện lắp"},
    {"vertical": "noi_that", "product": "bàn trang điểm", "focus": "diện tích phòng, ánh sáng gương, nhu cầu chứa đồ và chiều cao ghế", "proof": "kích thước, ngăn chứa, vật liệu và phương án giao lắp"},
]
INTENTS = [
    ("kham_pha_nhu_cau", "Tôi chưa biết chọn loại nào, bạn tư vấn giúp tôi với.", "làm rõ tiêu chí bắt buộc trước khi đề xuất"),
    ("so_sanh", "Tôi đang phân vân giữa hai mẫu, bạn giúp tôi so sánh công bằng nhé.", "lập bảng điểm giống và khác theo nhu cầu thực"),
    ("ngan_sach", "Ngân sách của tôi có hạn, có phương án nào hợp lý không?", "tách tiêu chí bắt buộc khỏi tính năng có thể nâng cấp sau"),
    ("gia_tri", "Mẫu này đắt hơn, vì sao tôi nên cân nhắc?", "giải thích giá trị bằng thông số, dịch vụ và chi phí sở hữu"),
    ("tra_gop", "Tôi muốn trả góp nhưng cần biết toàn bộ chi phí trước.", "minh bạch khoản trả trước, kỳ hạn, lãi và phí nếu có"),
    ("giao_lap", "Nhà tôi có không gian giao lắp hơi khó, có xử lý được không?", "xác minh kích thước, lối đi và khảo sát trước khi hứa"),
    ("bao_hanh", "Nếu có lỗi thì quy trình bảo hành như thế nào?", "nêu điều kiện, đầu mối, chứng từ và bước xử lý"),
    ("ton_kho", "Mẫu tôi cần có sẵn không, nếu hết thì sao?", "kiểm tra tồn kho thực tế trước và nêu rõ khác biệt của phương án thay thế"),
    ("tiet_kiem", "Tôi quan tâm chi phí sử dụng về lâu dài.", "ước tính có điều kiện dựa trên thông số công bố và thói quen dùng"),
    ("tuong_thich", "Sản phẩm này có hợp với đồ tôi đang dùng không?", "kiểm tra kích thước, kết nối, vật liệu hoặc nguồn điện trước"),
    ("combo", "Mua cùng các món liên quan có lợi hơn không?", "chỉ gợi ý combo khi tăng công năng hoặc giảm rủi ro tương thích"),
    ("doi_tra", "Nếu nhận hàng thấy không phù hợp thì tôi có đổi được không?", "xác nhận chính sách đổi trả bằng văn bản trước khi khách quyết định"),
    ("can_nhac", "Tôi muốn xem thêm rồi mới quyết định.", "tóm tắt lựa chọn và gửi thông tin để khách tự so sánh"),
    ("an_toan", "Tôi ưu tiên an toàn và độ bền hơn là nhiều tính năng.", "ưu tiên tiêu chuẩn, kết cấu, bảo hành và hướng dẫn sử dụng an toàn"),
    ("doanh_nghiep", "Tôi mua cho văn phòng hoặc căn hộ cho thuê, cần báo giá rõ ràng.", "xác định số lượng, tiến độ, điều kiện nghiệm thu và đầu mối bảo hành"),
]
PERSONAS = ["gia đình có con nhỏ", "người sống một mình", "cặp vợ chồng mới cưới", "người làm việc tại nhà", "khách thuê căn hộ", "chủ nhà mới nhận bàn giao", "gia đình nhiều thế hệ", "người lớn tuổi", "khách hàng ưu tiên tối giản", "người có thú cưng", "quản lý văn phòng", "chủ căn hộ cho thuê"]
CONTEXTS = ["căn hộ chung cư", "nhà phố", "phòng khách", "phòng ngủ", "phòng làm việc", "căn bếp", "studio nhỏ", "văn phòng", "căn hộ cho thuê", "nhà đang cải tạo", "không gian có trẻ nhỏ", "không gian có thú cưng", "phòng nhiều nắng", "khu vực ẩm", "khu vực có thang máy nhỏ", "nhà cần giao đúng lịch"]
CONSTRAINTS = ["cần giữ lối đi thông thoáng", "ưu tiên bảo hành rõ ràng", "muốn tránh phát sinh lúc giao lắp", "cần dùng ngay trong thời gian ngắn", "muốn so sánh tổng chi phí sở hữu", "cần kiểm soát ngân sách", "ưu tiên vệ sinh dễ", "cần tương thích đồ có sẵn", "không muốn thay đổi kết cấu nhà", "muốn kiểm tra hàng trước theo quy định", "cần hóa đơn rõ ràng", "cần tư vấn từ xa trước", "ưu tiên độ bền", "cần lắp ngoài giờ", "muốn có phương án thay thế", "cần người nhà cùng xác nhận"]
STAGES = ["mới tìm hiểu", "đang so sánh", "đã có mẫu tham khảo", "gần quyết định", "chờ đo đạc", "chờ hỏi ý kiến gia đình", "cần báo giá", "cần xác nhận lịch giao", "cần kiểm tra chính sách", "muốn thử trải nghiệm", "đang cân nhắc ngân sách", "muốn mua theo kế hoạch"]


def choose(index: int, values: list[Any]) -> tuple[Any, int]:
    return values[index % len(values)], index // len(values)


def build_record(index: int, seed: int) -> dict[str, Any]:
    cursor = index + seed
    product, cursor = choose(cursor, PRODUCTS)
    intent, cursor = choose(cursor, INTENTS)
    persona, cursor = choose(cursor, PERSONAS)
    context, cursor = choose(cursor, CONTEXTS)
    constraint, cursor = choose(cursor, CONSTRAINTS)
    stage, _ = choose(cursor, STAGES)
    intent_name, customer_request, strategy = intent
    prompt = f"Tình huống: Khách là {persona}, đang {stage} và tìm {product['product']} cho {context}. Khách {constraint}. Khách nói: {customer_request}"
    response = "\n".join([
        f"Dạ em hiểu anh/chị đang cần chọn {product['product']} phù hợp cho {context}, đồng thời {constraint}.",
        f"Để tư vấn đúng, em sẽ ưu tiên kiểm tra {product['focus']}.",
        f"Với trường hợp này, em đề xuất {strategy}. Anh/chị cho em thông tin về kích thước thực tế, ngân sách dự kiến và ưu tiên số một để em lọc 2–3 phương án phù hợp.",
        f"Trước khi chốt, em sẽ đối chiếu {product['proof']} và gửi rõ giá, phần giao lắp, bảo hành cùng các chi phí có thể phát sinh. Nếu phương án chưa đạt tiêu chí của anh/chị, mình tiếp tục so sánh hoặc dừng để chọn phương án khác; quyết định luôn thuộc về anh/chị.",
    ])
    identifier = hashlib.sha256(f"{seed}|{index}|{product['product']}|{intent_name}|{persona}|{context}|{constraint}|{stage}".encode("utf-8")).hexdigest()
    return {"id": identifier, "synthetic": True, "vertical": product["vertical"], "product_type": product["product"], "intent": intent_name, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}, {"role": "assistant", "content": response}]}


def select_validation(identifier: str, validation_ratio: float) -> bool:
    threshold = int(validation_ratio * 10000)
    return int(identifier[:8], 16) % 10000 < threshold


def generate_sales_playbook(config_path: str = "configs/synthetic_sales.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    count = int(config["count"])
    if count < 1:
        raise ValueError("count must be positive")
    validation_ratio = float(config["validation_ratio"])
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "sales_train.jsonl"
    validation_path = output_dir / "sales_validation.jsonl"
    train_temp = train_path.with_suffix(".jsonl.tmp")
    validation_temp = validation_path.with_suffix(".jsonl.tmp")
    train_count = 0
    validation_count = 0
    with train_temp.open("w", encoding="utf-8") as train_handle, validation_temp.open("w", encoding="utf-8") as validation_handle:
        for index in range(count):
            row = build_record(index, int(config["seed"]))
            destination = validation_handle if select_validation(row["id"], validation_ratio) else train_handle
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if destination is validation_handle:
                validation_count += 1
            else:
                train_count += 1
    train_temp.replace(train_path)
    validation_temp.replace(validation_path)
    manifest = {"name": "ethical-synthetic-sales-playbook", "created_at": datetime.now().astimezone().isoformat(), "total_examples": count, "train_examples": train_count, "validation_examples": validation_count, "seed": config["seed"], "synthetic": True, "verticals": ["noi_that", "dien_may"], "policies": ["no_deception", "no_fake_scarcity", "no_hidden_costs", "customer_autonomy"]}
    (output_dir / "sales_playbook_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
