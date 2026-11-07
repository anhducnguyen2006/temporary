from pathlib import Path

import pytest
from fastapi import HTTPException

from app.video_utils import save_and_validate_video


@pytest.mark.asyncio
async def test_invalid_video_type_is_rejected(tmp_path: Path):
    class Upload:
        filename = "sample.txt"
        content_type = "text/plain"

    with pytest.raises(HTTPException) as error:
        await save_and_validate_video(Upload(), tmp_path / "input.txt")
    assert error.value.status_code == 415

