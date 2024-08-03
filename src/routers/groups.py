from fastapi import APIRouter
from ..supabase import supabase
from pprint import pprint

router = APIRouter(
    prefix="/groups",
    tags=["Hello"],
)


@router.get("/pid/{profile_id}", description="Get group recommendations")
async def get_group_recommendations(profile_id: str):
    # fetch the profile of the user
    profile = (supabase
               .table("profiles")
               .select("*, student:students!students_profile_id_fkey(*)")
               .eq("id", profile_id)
               .single()
               .execute())

    # if profile is empty, return appropriate fastapi 404 error
    if not profile.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    # pprint(profile, indent=4)

    # anyway, we try to query the table of students, and their classes and subjects
    student = (supabase
                .table("students")
                .select("*, classes!classes_student_id_fkey(*, subjects(*))")
                .eq("profile_id", profile_id)
                .single()
                .execute())

    if not student.data:
        raise HTTPException(status_code=404, detail="Student not found")

    # anyway, we try to query the table of groups, and their classes and subjects
    groups = (supabase
                .table("groups")
                .select("*, course(*), group_members!group_members_group_id_fkey(*, profile:profiles!group_members_profile_id_fkey(*), student:students!group_members_student_id_fkey(*))")
                .eq("deleted", False)
                .neq("name", "")
                .execute())

    if not groups.data:
        raise HTTPException(status_code=404, detail="Groups not found")

    # filter the groups that the student is already a member of
    groups.data = [group for group in groups.data if not any(member["student"]["profile_id"] == profile_id for member in group["group_members"])]

    # TODO: implement the CBF algorithm

    # anyway let's do a basic recommendation for now
    # a group has these crucial infos: academic_year_id, year_level, avatar_url, student_no, course
    # the ranking will be based on the similarity of the academic_year_id, year_level, and course
    # to the current student / profile we're requesting this recommendation for     

    recommendations = []
    for group in groups.data:
        score = 0
        if group["academic_year_id"] == student.data["academic_year_id"]:
            score += 1
        if group["semester"]:
                score += 2
        if group["course"]:
            if group["course"] == student.data["course"]:
                score += 3
        if group["description"]:
            score += 1
        if group["avatar_url"]:
            score += 1

        recommendations.append({
            "group": group,
            "score": score
        })

        # pprint(recommendations, indent=4)

    # Sort recommendations by score in descending order
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    # Extract the group data from the recommendations
    recommendations = [recommendation['group'] for recommendation in recommendations]

    return {
        "groups": recommendations
    }