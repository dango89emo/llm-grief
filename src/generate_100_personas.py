#!/usr/bin/env python3
"""
Generate 100 diverse personas for grief diary data generation.
"""

import yaml
import random

# Lists for generating diverse personas
first_names = [
    "太郎", "花子", "健太", "美咲", "大樹", "愛美", "翔太", "さくら", "拓也", "由美",
    "優太", "結衣", "颯太", "美優", "蓮", "葵", "大和", "陽菜", "悠斗", "凛",
    "陸", "結菜", "湊", "咲良", "樹", "莉子", "蒼", "美羽", "碧", "心春",
    "律", "杏", "蒼太", "紬", "湊斗", "芽依", "颯", "結愛", "新", "愛莉",
    "朝陽", "莉緒", "碧斗", "陽葵", "晴", "彩乃", "悠真", "柚希", "結翔", "美月"
]

occupations = [
    "会社員", "エンジニア", "教師", "看護師", "医師", "デザイナー", "研究者", "営業職",
    "公務員", "弁護士", "美容師", "シェフ", "カウンセラー", "薬剤師", "建築士",
    "会計士", "記者", "編集者", "翻訳者", "介護士", "保育士", "歯科医", "獣医",
    "パイロット", "消防士", "警察官", "作家", "画家", "音楽家", "俳優", "YouTuber",
    "プログラマー", "データサイエンティスト", "マーケター", "コンサルタント", "起業家",
    "学生", "大学教授", "司書", "アーティスト", "フォトグラファー", "不動産営業",
    "ホテルスタッフ", "CA", "通訳", "栄養士", "トレーナー", "理学療法士", "社会福祉士"
]

important_others = [
    "母", "父", "祖母", "祖父", "配偶者", "恋人", "親友", "兄", "姉", "弟", "妹",
    "娘", "息子", "叔父", "叔母", "いとこ", "同僚", "メンター", "ペット（犬）", "ペット（猫）"
]

loss_events_templates = {
    "母": "最愛の母が病気で亡くなりました。長年の闘病の末、静かに息を引き取りました。",
    "父": "父が突然の心臓発作で亡くなりました。いつも元気だったのに、あまりにも急でした。",
    "祖母": "祖母が老衰で亡くなりました。穏やかな最期でしたが、寂しさは計り知れません。",
    "祖父": "祖父が脳卒中で急逝しました。最後まで現役で働いていた姿が思い出されます。",
    "配偶者": "最愛の配偶者を交通事故で亡くしました。突然の別れに心が引き裂かれそうです。",
    "恋人": "恋人が病気で亡くなりました。これから二人で築く予定だった未来が失われました。",
    "親友": "親友が自死を選びました。何も気づけなかった自分を責めています。",
    "兄": "兄が難病で亡くなりました。ずっと一緒にいてくれると思っていたのに。",
    "姉": "姉が出産時の合併症で亡くなりました。喜びが一転、深い悲しみに変わりました。",
    "弟": "弟がバイク事故で亡くなりました。もっと一緒に過ごせばよかったと後悔しています。",
    "妹": "妹が白血病で闘病の末に亡くなりました。最後まで明るく振る舞っていた姿が忘れられません。",
    "娘": "娘を病気で亡くしました。親が子を見送る辛さは言葉にできません。",
    "息子": "息子が事故で亡くなりました。まだこれからだったのに、受け入れられません。",
    "叔父": "叔父ががんで亡くなりました。いつも優しく接してくれた姿が目に浮かびます。",
    "叔母": "叔母が急病で亡くなりました。もっと話をしておけばよかったと思います。",
    "いとこ": "いとこが突然の病で亡くなりました。同年代で話が合い、大切な存在でした。",
    "同僚": "職場の同僚が過労で亡くなりました。一緒に働いていた仲間を失い、ショックです。",
    "メンター": "人生の師と仰いでいたメンターが亡くなりました。これからどう生きていけば良いのか。",
    "ペット（犬）": "15年一緒に過ごした愛犬が老衰で亡くなりました。家族同然の存在でした。",
    "ペット（猫）": "大切な愛猫が病気で亡くなりました。いつもそばにいてくれた温かさが恋しいです。",
}

def generate_persona(persona_id: int) -> dict:
    """Generate a single diverse persona."""
    name = random.choice(first_names)
    age = random.randint(20, 70)
    occupation = random.choice(occupations)
    important_other = random.choice(important_others)

    # Create description
    description = f"""あなたは{age}歳の{occupation}です。
名前は{name}です。
{important_other}はあなたにとって非常に大切な存在です。
日常生活や仕事、趣味、人間関係について、あなたの視点で日記を書いてください。"""

    # Get loss event template
    loss_event = loss_events_templates.get(
        important_other,
        f"{important_other}が亡くなりました。大切な存在を失った悲しみは計り知れません。"
    )

    return {
        "persona_id": persona_id,
        "name": name,
        "age": age,
        "occupation": occupation,
        "important_other": important_other,
        "description": description,
        "loss_event": loss_event,
    }

def main():
    """Generate 100 personas and save to YAML."""
    personas = [generate_persona(i + 1) for i in range(100)]

    data = {"personas": personas}

    with open("personas.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(personas)} personas and saved to personas.yaml")

    # Print summary
    print("\nSummary:")
    print(f"  Total personas: {len(personas)}")
    print(f"  Age range: {min(p['age'] for p in personas)}-{max(p['age'] for p in personas)}")
    print(f"  Unique occupations: {len(set(p['occupation'] for p in personas))}")
    print(f"  Unique important others: {len(set(p['important_other'] for p in personas))}")

if __name__ == "__main__":
    main()
