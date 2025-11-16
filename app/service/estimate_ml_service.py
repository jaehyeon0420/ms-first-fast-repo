# cost_estimation.py
# 두 모델의 결과를 결합하여 비용 견적 생성

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json
import os
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


import logging
import sys

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,  # stdout으로 출력
    format='%(asctime)s [%(levelname)s] %(message)s',
)

logger = logging.getLogger(__name__)

# 사용 예



# ============================================================================
# 1. 설정
# ============================================================================

current_file = Path(__file__).resolve()
APP_PATH = current_file.parent.parent  # service 상위 폴더 = app

# 모델 파일 path
MODEL_PATH = APP_PATH / "model" / "maskrcnn_model_final.pth"
MAPPING_PATH = APP_PATH / "model" / "damage_type_mapping.json"

# 출력 파일 path
RESULT_PATH = APP_PATH / "ml_outputs"

# app 폴더
# 추론 설정
CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 10

# 비용 설정 (px²당 원)
UNIT_COSTS = {
    "scratched": 100,      # 긁힘
    "crushed": 200,        # 찌그러짐
    "breakage": 300,       # 깨짐
    "separate": 150        # 도색손상/분리
}

# 손상 타입 매핑 (영어 -> 한글)
DAMAGE_TYPE_KR = {
    "scratched": "긁힘",
    "crushed": "찌그러짐",
    "breakage": "깨짐",
    "separate": "도색손상"
}


# ★★★ 시각화 최적화 설정 (작은 이미지용)
VISUALIZATION_CONFIG = {
    'box_width': 4,           # 박스 선 굵기 (더 굵게)
    'text_font_size': 16,     # 폰트 크기 (더 크게)
    'text_bg_padding': 6,     # 텍스트 배경 여백
    'upscale_factor': 1.5,    # 시각화용 업스케일 (1.0=원본, 2.0=2배)
    'text_offset_y': 25,      # 텍스트 Y 오프셋
}


# ============================================================================
# 색상 맵
# ============================================================================

def get_color_by_confidence(confidence):
    """신뢰도에 따라 색상 결정"""
    if confidence >= 0.5:
        return (0, 255, 0)      # 초록: 매우 높음 (50% 이상)
    elif confidence >= 0.4:
        return (255, 165, 0)    # 주황: 높음 (40-50%)
    elif confidence >= 0.3:
        return (255, 255, 0)    # 노랑: 중간 (30-40%)
    else:
        return (255, 0, 0)      # 빨강: 낮음 (30% 이하)

# ============================================================================
# 시스템 폰트 로드 (자동 선택)
# ============================================================================

def get_system_font(size=16):
    """운영체제별 시스템 폰트 자동 선택"""
    import platform
    
    system = platform.system()
    
    # Windows
    if system == 'Windows':
        font_paths = [
            r'C:\Windows\Fonts\arial.ttf',
            r'C:\Windows\Fonts\malgunbd.ttf',  # 한글
        ]
    # macOS
    elif system == 'Darwin':
        font_paths = [
            '/Library/Fonts/Arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
        ]
    # Linux
    else:
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        except:
            pass
    
    # 기본 폰트로 폴백
    return ImageFont.load_default()


# ============================================================================
# NMS (겹침 제거)
# ============================================================================
def nms_boxes(boxes, scores, threshold=0.5):
    """겹치는 박스 제거"""
    boxes = np.array(boxes)
    scores = np.array(scores)
    sorted_idx = np.argsort(-scores)
    keep = []
    
    while len(sorted_idx) > 0:
        current_idx = sorted_idx[0]
        keep.append(current_idx)
        
        if len(sorted_idx) == 1:
            break
        
        current_box = boxes[current_idx]
        other_boxes = boxes[sorted_idx[1:]]
        
        inter_x1 = np.maximum(current_box[0], other_boxes[:, 0])
        inter_y1 = np.maximum(current_box[1], other_boxes[:, 1])
        inter_x2 = np.minimum(current_box[2], other_boxes[:, 2])
        inter_y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        box1_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        box2_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area
        sorted_idx = sorted_idx[1:][iou < threshold]
    
    return keep


def estimate_torch_vision(cv_response_json) :
    
    # ============================================================================
    # damage_type_mapping.json 로드
    # ============================================================================

    try:
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"\n[MAPPING]")
        print(f"  Damage types: {list(mapping['damage_to_id'].keys())}")
    except Exception as e:
        logger.error(f"Mapping 로드 실패: {e}")
        

    # ============================================================================
    # maskrcnn_model_final.pth 로드
    # ============================================================================

    try:
        num_classes = len(mapping['damage_to_id'])
        model = maskrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        
        model_state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print(f"\n[MODEL]")
        print(f"  Classes: {num_classes}")
        print(f"  ✓ Model loaded")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        

    # ============================================================================
    # Custom Vision 모델 결과 로드
    # ============================================================================

    print(f"\n[MODEL 1 - CLASSIFICATION]")

    try:
        model1_data = cv_response_json
        
        # 확률 딕셔너리 생성
        model1_probs = {}
        for pred in model1_data['predictions']:
            tag = pred['tag']
            prob = pred['probability']
            model1_probs[tag] = prob
            print(f"  {tag}: {prob:.2%}")
        
    except Exception as e:
        logger.error(f"Classification 결과 로드 실패: {e}")
        

    # ============================================================================
    # Torch Vision Detection 모델 추론 시작
    # ============================================================================

    print(f"\n[MODEL 2 - DETECTION]")

    if not os.path.exists(cv_response_json['save_path']):
        logger.error(f"추론 대상 이미지 로드 실패: {e}")
        

    try:
        # 파손 이미지 로드
        image = Image.open(cv_response_json['save_path']).convert('RGB')
        image_width, image_height = image.size
        print(f"  Image size: {image_width}×{image_height}")
        
        image_tensor = torchvision.transforms.ToTensor()(image)
        
        # 추론 
        with torch.no_grad():
            predictions = model([image_tensor])
        
        prediction = predictions[0]
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        print(f"  Raw detections: {len(labels)}")
        
        # 신뢰도 필터링
        valid_mask = scores >= CONFIDENCE_THRESHOLD
        boxes_filtered = boxes[valid_mask]
        labels_filtered = labels[valid_mask]
        scores_filtered = scores[valid_mask]
        
        print(f"  After confidence filter: {len(labels_filtered)}")
        
        # NMS
        if len(boxes_filtered) > 0:
            keep_idx = nms_boxes(boxes_filtered, scores_filtered, NMS_THRESHOLD)
            boxes_nms = boxes_filtered[keep_idx]
            labels_nms = labels_filtered[keep_idx]
            scores_nms = scores_filtered[keep_idx]
        else:
            boxes_nms = boxes_filtered
            labels_nms = labels_filtered
            scores_nms = scores_filtered
        
        print(f"  After NMS: {len(labels_nms)}")
        
        # 상위 N개
        if len(labels_nms) > MAX_DETECTIONS:
            top_idx = np.argsort(-scores_nms)[:MAX_DETECTIONS]
            boxes_final = boxes_nms[top_idx]
            labels_final = labels_nms[top_idx]
            scores_final = scores_nms[top_idx]
        else:
            boxes_final = boxes_nms
            labels_final = labels_nms
            scores_final = scores_nms
        
        print(f"  Final detections: {len(labels_final)}")
        
        # Model 2 결과 정리
        model2_results = []
        for label, score, box in zip(labels_final, scores_final, boxes_final):
            damage_type = mapping['id_to_damage'].get(str(label), 'Unknown')
            x1, y1, x2, y2 = box.astype(int)
            area = (x2 - x1) * (y2 - y1)
            
            model2_results.append({
                "type": damage_type.lower(),
                "area": int(area),
                "confidence": float(score),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })
        
    except Exception as e:
        logger.error(f"Detection 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        

    # ============================================================================
    # 비용 계산
    # ============================================================================

    print(f"\n[COST CALCULATION]")

    results = []

    idx = 1
    
    for region in model2_results:
        damage_type = region["type"]
        area = region["area"]
        model2_conf = region["confidence"]
        
        # Model 1 확률 가져오기 (없으면 0.5 기본값)
        model1_prob = model1_probs.get(damage_type, 0.5)
        
        # 단위 비용 가져오기
        unit_cost = UNIT_COSTS.get(damage_type, 100)
        
        # 기본 비용
        base_cost = area * unit_cost
        
        # 최소 신뢰도
        min_conf = min(model2_conf, model1_prob)
        
        # 불확실성 계산
        model2_unc = 1 - model2_conf
        model1_unc = 1 - model1_prob
        combined_unc = math.sqrt(model2_unc**2 + model1_unc**2)
        
        # 최대 신뢰도
        max_conf = min(1.0, min_conf + combined_unc)
        
        # 추천 신뢰도 (평균)
        rec_conf = (model2_conf + model1_prob) / 2
        
        # 비용 계산
        min_cost = base_cost * min_conf
        max_cost = base_cost * max_conf
        rec_cost = base_cost * rec_conf
        
        damage_type_kr = DAMAGE_TYPE_KR.get(damage_type, damage_type)
        
        results.append({
            "id" : idx,
            "type": damage_type,
            "type_kr": damage_type_kr,
            "area": area,
            "bbox": region["bbox"],
            "base_cost": int(base_cost),
            "min_cost": int(min_cost),
            "max_cost": int(max_cost),
            "recommended_cost": int(rec_cost),
            "confidence": {
                "min": round(min_conf, 4),
                "max": round(max_conf, 4),
                "recommended": round(rec_conf, 4),
                "model1_prob": round(model1_prob, 4),
                "model2_conf": round(model2_conf, 4)
            }
        })
        
        idx += 1
        
    # ====================================================================
    # 이미지에 영역 표시 및 업스케일링 후, 저장
    # ====================================================================
    
    print(f"\n[VISUALIZATION]")
    
    # 업스케일링 (작은 이미지 읽기 쉽게)
    upscale_factor = VISUALIZATION_CONFIG['upscale_factor']
    new_width = int(image_width * upscale_factor)
    new_height = int(image_height * upscale_factor)
    
    image_viz = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"  Upscaled to: {new_width}×{new_height} (factor: {upscale_factor}x)")
    
    draw = ImageDraw.Draw(image_viz)
    
    font = ''
    try :
        font = get_system_font(VISUALIZATION_CONFIG['text_font_size'])
    except Exception as e:
        logger.error(f"시스템 폰트 로드 get_system_font() 실패: {e}")
        
    
    # 박스 및 텍스트 그리기
    for i, (label, score, box) in enumerate(zip(labels_final, scores_final, boxes_final)):
        damage_type = mapping['id_to_damage'].get(str(label), 'Unknown')
        confidence = score * 100
        
        # 스케일 적용
        x1, y1, x2, y2 = (box * upscale_factor).astype(int)
        
        # 색상
        color = get_color_by_confidence(score)
        
        # 박스 색상 정보 #000000 형식으로 변환하여 저장
        results[i]['color'] = "#{:02X}{:02X}{:02X}".format(*color)
        
        
        # ★ 박스 그리기 (굵게)
        box_width = VISUALIZATION_CONFIG['box_width']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        
        # ★ 텍스트 준비
        label_text = f"{damage_type} {confidence:.1f}%"
        
        # ★ 텍스트 배경 (명시적)
        text_offset_y = VISUALIZATION_CONFIG['text_offset_y']
        text_y = max(0, y1 - text_offset_y)  # 위에 표시
        
        bbox = draw.textbbox((x1, text_y), label_text, font=font)
        padding = VISUALIZATION_CONFIG['text_bg_padding']
        
        # 배경 박스
        bg_bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ]
        draw.rectangle(bg_bbox, fill=color)
        
        # 텍스트 (흰색, 명시적)
        draw.text((x1, text_y), label_text, fill=(255, 255, 255), font=font)
        
        # 박스 번호 (우측 상단)
        number_text = f"#{i+1}"
        draw.text((x2 - 40, y1 + 5), number_text, fill=color, font=font)
    
    # 이미지 저장
    name, ext = os.path.splitext(cv_response_json['image_file'])
    jpg_filename = f"{name}_image.jpg"
    viz_path = RESULT_PATH / jpg_filename
    image_viz.save(viz_path)
    print(f"  ✓ Saved (upscaled): {viz_path}")
    
    # ============================================================================
    # 결과 출력 (콘솔)
    # ============================================================================

    print("\n=== 영역별 상세 견적 ===\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['type_kr']} ({r['area']}px²)")
        print(f"   기본 비용: {r['base_cost']:,}원")
        print(f"   신뢰도 범위: {r['confidence']['min']:.1%} ~ {r['confidence']['max']:.1%}")
        print(f"   비용 범위: {r['min_cost']:,}원 ~ {r['max_cost']:,}원")
        print(f"   추천 비용: {r['recommended_cost']:,}원")
        print()

    # 총계
    total_min = sum(r["min_cost"] for r in results)
    total_max = sum(r["max_cost"] for r in results)
    total_rec = sum(r["recommended_cost"] for r in results)

    print("=== 총 견적 ===")
    print(f"최소 견적: {total_min:,}원")
    print(f"최대 견적: {total_max:,}원")
    print(f"추천 견적: {total_rec:,}원 ★")

    # ============================================================================
    # inference_result.json 저장 (기존 포맷)
    # ============================================================================

    inference_result = {
        "image_size": [image_width, image_height],
        "total_detections": len(labels),
        "filtered_detections": len(labels_final),
        "detections": []
    }

    for i, (label, score, box) in enumerate(zip(labels_final, scores_final, boxes_final)):
        damage_type = mapping['id_to_damage'].get(str(label), 'Unknown')
        x1, y1, x2, y2 = box.astype(int)
        
        inference_result["detections"].append({
            "id": i + 1,
            "damage_type": damage_type,
            "confidence": float(score),
            "confidence_percent": round(float(score) * 100, 1),
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "area": int((x2 - x1) * (y2 - y1))
            }
        })

    try:
        # inference.json 생성
        name, ext = os.path.splitext(cv_response_json['image_file'])
        inference_json_filename = f"{name}_inference.json"
        
        with open(RESULT_PATH / inference_json_filename, 'w', encoding='utf-8') as f:
            json.dump(inference_result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved inference result: {inference_json_filename}")
    except Exception as e:
        logger.error(f"Failed to save inference result : {e}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # cost_result.json 저장
    # ============================================================================

    cost_result = {
        "image_file": os.path.basename(cv_response_json['save_path']),
        "image_size": [image_width, image_height],
        "model1_probabilities": {
            DAMAGE_TYPE_KR.get(k, k): round(v, 4) 
            for k, v in model1_probs.items()
        },
        "total_detections": len(results),
        "regions": results,
        "summary": {
            "total_min_cost": int(total_min),
            "total_max_cost": int(total_max),
            "total_recommended_cost": int(total_rec)
        },
        "unit_costs": {
            DAMAGE_TYPE_KR.get(k, k): v 
            for k, v in UNIT_COSTS.items()
        }
    }

    # cost.json 생성
    name, ext = os.path.splitext(cv_response_json['image_file'])
    cost_json_filename = f"{name}_cost.json"
    
    try:
        with open(RESULT_PATH / cost_json_filename, 'w', encoding='utf-8') as f:
            json.dump(cost_result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved cost result: {cost_json_filename}")
    except Exception as e:
        logger.error(f"Failed to save cost result : {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ Cost estimation completed!")