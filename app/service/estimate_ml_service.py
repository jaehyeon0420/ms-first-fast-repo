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
import platform
from pathlib import Path
import sys


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
OVERLAP_IOU_THRESHOLD = 0.3  # 겹침 판단 기준 (30% 이상 겹치면 겹친 것으로 판단)
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
# 겹침 제거 (더 큰 영역 우선)
# ============================================================================

def calculate_iou(box1, box2):
    """두 박스의 IoU 계산"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 교집합 영역
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 합집합 영역
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def filter_overlapping_boxes(boxes, scores, labels, iou_threshold=0.3):
    """
    겹치는 박스들 중 더 큰 영역을 가진 박스만 유지
    
    Args:
        boxes: numpy array of boxes [N, 4]
        scores: numpy array of scores [N]
        labels: numpy array of labels [N]
        iou_threshold: IoU 임계값 (이 값 이상이면 겹친 것으로 판단)
    
    Returns:
        keep_indices: 유지할 박스의 인덱스 리스트
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 각 박스의 면적 계산
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 처리할 박스들의 인덱스 (처음엔 모든 박스)
    remaining_indices = list(range(len(boxes)))
    keep_indices = []
    
    while remaining_indices:
        # 남은 박스들 중 가장 큰 영역을 가진 박스 선택
        max_area_idx = max(remaining_indices, key=lambda i: areas[i])
        keep_indices.append(max_area_idx)
        
        # 선택된 박스와 겹치는 박스들 찾기
        current_box = boxes[max_area_idx]
        overlapping_indices = []
        
        for idx in remaining_indices:
            if idx == max_area_idx:
                continue
            
            iou = calculate_iou(current_box, boxes[idx])
            
            if iou >= iou_threshold:
                overlapping_indices.append(idx)
        
        # 선택된 박스와 겹치는 박스들을 제거 목록에 추가
        remaining_indices.remove(max_area_idx)
        for idx in overlapping_indices:
            if idx in remaining_indices:
                remaining_indices.remove(idx)
    
    return keep_indices

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
    """운영체제별 한글 지원 폰트 자동 선택"""
    system = platform.system()
    
    if system == 'Windows':
        font_paths = [
            r'C:\Windows\Fonts\malgun.ttf',      # 맑은 고딕
            r'C:\Windows\Fonts\malgunbd.ttf',    # 맑은 고딕 Bold
            r'C:\Windows\Fonts\gulim.ttc',       # 굴림
            r'C:\Windows\Fonts\batang.ttc',      # 바탕
            r'C:\Windows\Fonts\NanumGothic.ttf', # 나눔고딕
            r'C:\Windows\Fonts\arial.ttf',       # Arial (fallback)
        ]
    elif system == 'Darwin':
        font_paths = [
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/Library/Fonts/Arial.ttf',
        ]
    else:
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
    
    # 폰트 찾기
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size)
                print(f"  Using font: {os.path.basename(font_path)}")
                return font
        except Exception as e:
            continue
    
    # 모든 폰트 실패 시 기본 폰트
    print(f"  Warning: Using default font (Korean may not display correctly)")
    return ImageFont.load_default()


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
        sys.stderr.write(f'Mapping 로드 실패 {e}')  
        #logger.error(f"Mapping 로드 실패: {e}")
        

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
        sys.stderr.write(f'모델 로드 실패 {e}')  
        
        #logger.error(f"모델 로드 실패: {e}")
        

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
        sys.stderr.write(f'Classification 결과 로드 실패 {e}')  
        #logger.error(f"Classification 결과 로드 실패: {e}")
        

    # ============================================================================
    # Torch Vision Detection 모델 추론 시작
    # ============================================================================

    print(f"\n[MODEL 2 - DETECTION]")

    if not os.path.exists(APP_PATH / "uploads" / cv_response_json['image_file'][:8] / cv_response_json['image_file']):
        sys.stderr.write(f'추론 대상 이미지 로드 실패')  
        #logger.error(f"추론 대상 이미지 로드 실패")
        
    # Model 2 결과 정리
    model2_results = []
        
    try:
        # 파손 이미지 로드
        image = Image.open(APP_PATH / "uploads" / cv_response_json['image_file'][:8] / cv_response_json['image_file']).convert('RGB')
        image_width, image_height = image.size
        print(f"  Image size: {image_width}×{image_height}")
        
        image_tensor = torchvision.transforms.ToTensor()(image)
        
        # 추론 
        model.eval()
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
        
        # 2단계: 겹치는 박스 제거 (더 큰 영역 우선)
        if len(boxes_filtered) > 0:
            keep_idx = filter_overlapping_boxes(
                boxes_filtered, 
                scores_filtered, 
                labels_filtered, 
                OVERLAP_IOU_THRESHOLD
            )
            boxes_final = boxes_filtered[keep_idx]
            labels_final = labels_filtered[keep_idx]
            scores_final = scores_filtered[keep_idx]
            
            print(f"  After overlap removal (larger area priority): {len(labels_final)}")
        else:
            boxes_final = boxes_filtered
            labels_final = labels_filtered
            scores_final = scores_filtered
        
        # 3단계: 상위 N개로 제한
        if len(labels_final) > MAX_DETECTIONS:
            areas = (boxes_final[:, 2] - boxes_final[:, 0]) * (boxes_final[:, 3] - boxes_final[:, 1])
            top_idx = np.argsort(-areas)[:MAX_DETECTIONS]
            boxes_final = boxes_final[top_idx]
            labels_final = labels_final[top_idx]
            scores_final = scores_final[top_idx]
            print(f"  Limited to top {MAX_DETECTIONS} by area")
        
        print(f"  Final detections: {len(labels_final)}")
        
        
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
            
            
        # 영역 크기 순으로 정렬
        model2_results.sort(key=lambda x: x["area"], reverse=True)
        
    except Exception as e:
        sys.stderr.write(f'Detection 추론 실패: {e}')  
        #logger.error(f"Detection 추론 실패: {e}")
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
        sys.stderr.write(f'시스템 폰트 로드 실패: {e}')  
        print('시스템 폰트 로드 실패')
        
    total_min = sum(r["min_cost"] for r in results)
    total_max = sum(r["max_cost"] for r in results)
    total_rec = sum(r["recommended_cost"] for r in results)
    
    # 박스 및 텍스트 그리기
    for i, result in enumerate(results):
        bbox = result["bbox"]
        confidence = result["confidence"]["model2_conf"]
        damage_type_kr = result["type_kr"]
        area = result["area"]
        
        # 스케일 적용
        x1 = int(bbox["x1"] * upscale_factor)
        y1 = int(bbox["y1"] * upscale_factor)
        x2 = int(bbox["x2"] * upscale_factor)
        y2 = int(bbox["y2"] * upscale_factor)
        
        # 색상
        color = get_color_by_confidence(confidence)
        results[i]['color'] = '#{:02x}{:02x}{:02x}'.format(*color)
        
        # 박스 그리기
        box_width = VISUALIZATION_CONFIG['box_width']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        
        # 텍스트 준비
        label_text = f"{damage_type_kr} {confidence*100:.1f}%"
        cost_text = f"{result['recommended_cost']:,}원"
        
        # 텍스트 배경
        text_offset_y = VISUALIZATION_CONFIG['text_offset_y']
        text_y = max(0, y1 - text_offset_y)
        
        bbox_text = draw.textbbox((x1, text_y), label_text, font=font)
        padding = VISUALIZATION_CONFIG['text_bg_padding']
        
        # 배경 박스
        bg_bbox = [
            bbox_text[0] - padding,
            bbox_text[1] - padding,
            bbox_text[2] + padding,
            bbox_text[3] + padding
        ]
        draw.rectangle(bg_bbox, fill=color)
        
        # 텍스트 (흰색)
        draw.text((x1, text_y), label_text, fill=(255, 255, 255), font=font)
        
        # 비용 텍스트 (박스 내부 하단)
        cost_y = y2 - 25
        draw.text((x1 + 5, cost_y), cost_text, fill=color, font=font)
        
        # 박스 번호 (우측 상단)
        number_text = f"#{i+1}"
        draw.text((x2 - 40, y1 + 5), number_text, fill=color, font=font)
    
    
    # 총 견적 정보 추가 (이미지 상단)
    summary_text = f"총 견적: {total_rec:,}원 ({len(results)}개 손상)"
    summary_bg_bbox = draw.textbbox((10, 10), summary_text, font=font)
    draw.rectangle([
        summary_bg_bbox[0] - 10,
        summary_bg_bbox[1] - 10,
        summary_bg_bbox[2] + 10,
        summary_bg_bbox[3] + 10
    ], fill=(0, 0, 0))
    draw.text((10, 10), summary_text, fill=(255, 255, 255), font=font)
    
    # 업스케일 버전저장       
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
        sys.stderr.write(f'failed to save result: {e}')  
        print('failed to save result')
        import traceback
        traceback.print_exc()

    # ============================================================================
    # cost_result.json 저장
    # ============================================================================

    cost_result = {
        "image_file": os.path.basename(APP_PATH / "uploads" / cv_response_json['image_file'][:8] / cv_response_json['image_file']),
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
        sys.stderr.write(f'failed to save cost result: {e}')  
        print('failed to save cost result ')
        import traceback
        traceback.print_exc()

    print(f"\n✅ Cost estimation completed!")