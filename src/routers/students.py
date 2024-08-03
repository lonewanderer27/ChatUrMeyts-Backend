from fastapi import APIRouter, HTTPException
from ..supabase import supabase
from pprint import pprint

router = APIRouter(
    prefix="/students",
    tags=["Hello"],
)

@router.get("/pid/{profile_id}", description="Get student recommendations")
async def get_student_recommendations(profile_id: str):
    # fetch the profile of the user
    profile = (supabase
               .table("profiles")
               .select("*")
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

    # pprint(student, indent=4)

    if not student.data:
        raise HTTPException(status_code=404, detail="Student not found")

    
    # query the table of students
    students = (supabase
                    .table("students")
                    .select("*, profile:profiles!students_profile_id_fkey(*), course:courses(*)")
                    .neq("profile_id", profile_id)
                    .eq("verified", True)
                    .order("full_name")
                    .execute())

    # pprint(students, indent=4)

    # TODO: implement the CBF algorithm

    # anyway let's do a basic recommendation for now
    # each student has these crucial infos: academic_year_id, year_level, avatar_url, student_no, course
    # the ranking will be based on the similarity of the academic_year_id, year_level, and course
    # to the student we're requesting this recommendation for     

    recommendations = []
    for other_student in students.data:
        score = 0
        if other_student['academic_year_id'] == student.data['academic_year_id']:
            score += 1
        if other_student['year_level'] == student.data['year_level']:
            score += 2
        if other_student['course'] == student.data['course']:
            score += 3
        # if the other student has description, add them a point
        if other_student['description']:
            score += 1
        # if the other student has avatar_url, add them a point
        if other_student['avatar_url']:
            score += 1
        else:
            score -= 3
        # if the other student has student_no, add them a point
        if other_student['student_no']:
            score += 1
        
        recommendations.append({
            "student": other_student,
            "similarity_score": score
        })

        pprint(recommendations, indent=4)

    # Sort recommendations by similarity score in descending order
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)

    # Extract the student data from the recommendations
    recommendations = [recommendation['student'] for recommendation in recommendations]
    
    return {
        "students": recommendations,
    }
