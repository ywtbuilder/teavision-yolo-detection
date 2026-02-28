# -*- coding: utf-8 -*-
"""
TeaVision V13 | 数据访问层 (DAO)

管理检测记录的持久化存储与查询。

=== 数据库表结构 ===
1. detection_records — 检测记录主表
2. detection_objects — 检测目标详情表

=== 对外接口 ===
写入：
- save_detection_record()    → 保存一次检测的完整记录

查询：
- get_total_stats()          → 系统总览统计
- get_daily_trend()          → 每日检测趋势
- get_variety_distribution() → 品种分布
- get_hourly_distribution()  → 小时分布
- get_detection_history()    → 历史记录列表
- get_weekly_heatmap()       → 一周热力图

维护：
- clear_old_records()        → 清理旧记录
- get_db_stats()             → 数据库自身统计
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import threading

# 数据库文件路径（相对于 dao 目录）
DB_DIR = Path(__file__).parent / "data"
DB_PATH = DB_DIR / "detections.db"

# 线程锁，确保数据库操作线程安全
_db_lock = threading.Lock()


# ==================== 连接管理 ====================

def get_connection():
    """获取数据库连接"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # 返回字典格式
    return conn


def init_db():
    """初始化数据库，创建表结构"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        # 检测记录主表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_name VARCHAR(255),
                image_width INTEGER,
                image_height INTEGER,
                model_name VARCHAR(100),
                total_objects INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                inference_time_ms REAL DEFAULT 0,
                detection_type VARCHAR(20) DEFAULT 'image',
                conf_threshold REAL DEFAULT 0.25,
                iou_threshold REAL DEFAULT 0.45
            )
        ''')

        # 检测目标详情表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id INTEGER NOT NULL,
                class_id INTEGER,
                class_name VARCHAR(100),
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                FOREIGN KEY (record_id) REFERENCES detection_records(id) ON DELETE CASCADE
            )
        ''')

        # 创建索引优化查询
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_time ON detection_records(detection_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_type ON detection_records(detection_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_object_class ON detection_objects(class_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_object_record ON detection_objects(record_id)')

        conn.commit()
        conn.close()


# ==================== 写入操作 ====================

def save_detection_record(
    image_name: str,
    image_size: tuple,
    model_name: str,
    total_objects: int,
    avg_confidence: float,
    inference_time_ms: float,
    detection_type: str = 'image',
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    objects: Optional[List[Dict]] = None
) -> int:
    """
    保存检测记录

    Args:
        image_name:       图片名称
        image_size:       图片尺寸 (width, height)
        model_name:       模型名称
        total_objects:    检测到的目标总数
        avg_confidence:   平均置信度
        inference_time_ms: 推理耗时(毫秒)
        detection_type:   检测类型 (image/camera/batch)
        conf_threshold:   置信度阈值
        iou_threshold:    IoU阈值
        objects:          检测目标列表

    Returns:
        record_id: 记录ID
    """
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        # 插入检测记录
        cursor.execute('''
            INSERT INTO detection_records
            (image_name, image_width, image_height, model_name, total_objects,
             avg_confidence, inference_time_ms, detection_type, conf_threshold, iou_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_name,
            image_size[0] if image_size else 0,
            image_size[1] if image_size else 0,
            model_name,
            total_objects,
            avg_confidence,
            inference_time_ms,
            detection_type,
            conf_threshold,
            iou_threshold
        ))

        record_id = cursor.lastrowid

        # 插入检测目标详情
        if objects:
            for obj in objects:
                bbox = obj.get('bbox', [0, 0, 0, 0])
                cursor.execute('''
                    INSERT INTO detection_objects
                    (record_id, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record_id,
                    obj.get('class_id', 0),
                    obj.get('class_name', ''),
                    obj.get('confidence', 0),
                    int(bbox[0]) if len(bbox) > 0 else 0,
                    int(bbox[1]) if len(bbox) > 1 else 0,
                    int(bbox[2]) if len(bbox) > 2 else 0,
                    int(bbox[3]) if len(bbox) > 3 else 0
                ))

        conn.commit()
        conn.close()

        return record_id


# ==================== 统计查询 ====================

def get_total_stats() -> Dict[str, Any]:
    """获取总体统计数据"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        # 总检测次数
        cursor.execute("SELECT COUNT(*) FROM detection_records")
        total_detections = cursor.fetchone()[0]

        # 今日检测次数
        cursor.execute("""
            SELECT COUNT(*) FROM detection_records
            WHERE DATE(detection_time, 'localtime') = DATE('now', 'localtime')
        """)
        today_detections = cursor.fetchone()[0]

        # 平均置信度
        cursor.execute("SELECT AVG(avg_confidence) FROM detection_records WHERE total_objects > 0")
        avg_confidence = cursor.fetchone()[0] or 0

        # 平均推理时间
        cursor.execute("SELECT AVG(inference_time_ms) FROM detection_records")
        avg_inference_time = cursor.fetchone()[0] or 0

        # 已录入茶叶品种数量（与前端档案库一致）
        varieties_count = 175

        # 总检测目标数
        cursor.execute("SELECT SUM(total_objects) FROM detection_records")
        total_objects = cursor.fetchone()[0] or 0

        conn.close()

        return {
            'total_detections': total_detections,
            'today_detections': today_detections,
            'avg_confidence': avg_confidence,
            'avg_inference_time': avg_inference_time,
            'varieties_count': varieties_count,
            'total_objects': total_objects
        }


def get_daily_trend(days: int = 7) -> Dict[str, List]:
    """获取每日检测趋势数据"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        dates = []
        detections = []
        accuracy = []
        processing_time = []

        for i in range(days - 1, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append((datetime.now() - timedelta(days=i)).strftime("%m/%d"))

            # 当日检测次数
            cursor.execute("""
                SELECT COUNT(*) FROM detection_records
                WHERE DATE(detection_time, 'localtime') = ?
            """, (date,))
            count = cursor.fetchone()[0]
            detections.append(count)

            # 当日平均置信度
            cursor.execute("""
                SELECT AVG(avg_confidence) FROM detection_records
                WHERE DATE(detection_time, 'localtime') = ? AND total_objects > 0
            """, (date,))
            acc = cursor.fetchone()[0]
            accuracy.append(acc if acc else 0)

            # 当日平均推理时间
            cursor.execute("""
                SELECT AVG(inference_time_ms) FROM detection_records
                WHERE DATE(detection_time, 'localtime') = ?
            """, (date,))
            time_ms = cursor.fetchone()[0]
            processing_time.append(time_ms if time_ms else 0)

        conn.close()

        return {
            'dates': dates,
            'detections': detections,
            'accuracy': accuracy,
            'processing_time': processing_time
        }


def get_variety_distribution() -> Dict[str, int]:
    """获取品种检测分布"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT class_name, COUNT(*) as cnt
            FROM detection_objects
            WHERE class_name IS NOT NULL AND class_name != ''
            GROUP BY class_name
            ORDER BY cnt DESC
        """)

        result = {row['class_name']: row['cnt'] for row in cursor.fetchall()}
        conn.close()

        return result


def get_hourly_distribution() -> Dict[str, List]:
    """获取小时分布数据"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        hours = list(range(24))
        counts = [0] * 24

        cursor.execute("""
            SELECT strftime('%H', detection_time, 'localtime') as hour, COUNT(*) as cnt
            FROM detection_records
            GROUP BY hour
        """)

        for row in cursor.fetchall():
            hour = int(row['hour'])
            counts[hour] = row['cnt']

        conn.close()

        return {'hours': hours, 'counts': counts}


def get_detection_history(limit: int = 20, status_filter: str = "全部", variety_filter: str = "全部") -> List[Dict]:
    """获取检测历史记录"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        # 基础查询
        query = """
            SELECT
                r.id,
                datetime(r.detection_time, 'localtime') as detection_time,
                r.image_name,
                r.total_objects,
                r.avg_confidence,
                r.inference_time_ms,
                GROUP_CONCAT(DISTINCT o.class_name) as varieties
            FROM detection_records r
            LEFT JOIN detection_objects o ON r.id = o.record_id
        """

        conditions = []
        params = []

        # 品种筛选
        if variety_filter != "全部":
            conditions.append("o.class_name = ?")
            params.append(variety_filter)

        # 状态筛选
        if status_filter == "成功":
            conditions.append("r.total_objects > 0")
        elif status_filter == "失败":
            conditions.append("r.total_objects = 0")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY r.id ORDER BY r.detection_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        history = []
        for row in cursor.fetchall():
            status = "成功" if row['total_objects'] > 0 else "失败"
            varieties = row['varieties'] if row['varieties'] else "无"

            history.append({
                'id': row['id'],
                '时间': row['detection_time'][:16] if row['detection_time'] else '',
                '品种': varieties.split(',')[0] if varieties != "无" else "未检测到",
                '置信度': f"{row['avg_confidence']:.1%}" if row['avg_confidence'] else "-",
                '耗时': f"{row['inference_time_ms']:.0f}ms" if row['inference_time_ms'] else "-",
                '状态': status,
                '目标数': row['total_objects']
            })

        conn.close()

        return history


def get_weekly_heatmap() -> List[List[int]]:
    """获取一周热力图数据 (7天 x 24小时)"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        # 初始化 7x24 矩阵
        heatmap = [[0 for _ in range(24)] for _ in range(7)]

        cursor.execute("""
            SELECT
                strftime('%w', detection_time, 'localtime') as weekday,
                strftime('%H', detection_time, 'localtime') as hour,
                COUNT(*) as cnt
            FROM detection_records
            WHERE detection_time >= datetime('now', '-7 days')
            GROUP BY weekday, hour
        """)

        for row in cursor.fetchall():
            weekday = int(row['weekday'])
            hour = int(row['hour'])
            heatmap[weekday][hour] = row['cnt']

        conn.close()

        return heatmap


# ==================== 维护操作 ====================

def clear_old_records(days: int = 30):
    """清理指定天数前的旧记录"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM detection_records
            WHERE detection_time < datetime('now', ?)
        """, (f'-{days} days',))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count


def get_db_stats() -> Dict[str, Any]:
    """获取数据库统计信息"""
    with _db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM detection_records")
        records_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM detection_objects")
        objects_count = cursor.fetchone()[0]

        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

        cursor.execute("SELECT MIN(detection_time) FROM detection_records")
        earliest = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(detection_time) FROM detection_records")
        latest = cursor.fetchone()[0]

        conn.close()

        return {
            'records_count': records_count,
            'objects_count': objects_count,
            'db_size_mb': db_size / (1024 * 1024),
            'earliest_record': earliest,
            'latest_record': latest
        }


# 初始化数据库（模块加载时自动执行）
init_db()
